"""
This module expands the functionality of the CtrlAviary with
new dynamics that can account for assymetric moments of inertia
"""

import numpy as np
import pybullet as p
import xml.etree.ElementTree as etxml
import pkg_resources
import os
from datetime import datetime
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class VariationAviary(CtrlAviary):
    """Multi-Drone environment class for control of asymmetric drones"""

    #######################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 vision_attributes=False,
                 dynamics_attributes=False,
                 output_folder='results',
                 wind_force_vector = [0.,0.,0.]
                 ):
        """Initialization of a generic aviary environment.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        vision_attributes : bool, optional
            Whether to allocate the attributes needed by vision-based aviary subclasses.
        dynamics_attributes : bool, optional
            Whether to allocate the attributes needed by subclasses accepting thrust and torques inputs.

        """
        #### Constants #############################################
        self.G = 9.8
        self.RAD2DEG = 180/np.pi
        self.DEG2RAD = np.pi/180
        self.SIM_FREQ = freq
        self.TIMESTEP = 1./self.SIM_FREQ
        self.AGGR_PHY_STEPS = aggregate_phy_steps
        #### Parameters ############################################
        self.NUM_DRONES = num_drones
        self.NEIGHBOURHOOD_RADIUS = neighbourhood_radius
        #### Options ###############################################
        self.DRONE_MODEL = drone_model
        self.GUI = gui
        self.RECORD = record
        self.PHYSICS = physics
        self.OBSTACLES = obstacles
        self.USER_DEBUG = user_debug_gui
        self.URDF = self.DRONE_MODEL.value + ".urdf"
        self.OUTPUT_FOLDER = output_folder
        #### Load the drone properties from the .urdf file #########
        self.M, \
        self.L, \
        self.THRUST2WEIGHT_RATIO, \
        self.J, \
        self.J_INV, \
        self.KF, \
        self.KM, \
        self.COLLISION_H,\
        self.COLLISION_R, \
        self.COLLISION_Z_OFFSET, \
        self.MAX_SPEED_KMH, \
        self.GND_EFF_COEFF, \
        self.PROP_RADIUS, \
        self.DRAG_COEFF, \
        self.DW_COEFF_1, \
        self.DW_COEFF_2, \
        self.DW_COEFF_3, \
        self.COM_XYZ = self._parseURDFParameters()
        print("[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:\n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, ixy {:f}, ixz {:f}, iyy {:f}, iyz {:f}, izz {:f},\n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f},\n[INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f},\n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}, com_xyz {}".format(
            self.M, self.L, self.J[0,0], self.J[0,1], self.J[0,2], self.J[1,1], self.J[1,2], self.J[2,2], self.KF, self.KM, self.THRUST2WEIGHT_RATIO, self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS, self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1, self.DW_COEFF_2, self.DW_COEFF_3, self.COM_XYZ))
        #### Compute constants #####################################
        self.GRAVITY = self.G * self.M
        # self.WIND = np.array(wind_acc_vector)
        self.WIND_FORCE = np.array(wind_force_vector)
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4*self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.KF))
        self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        elif self.DRONE_MODEL in [DroneModel.CF2P, DroneModel.HB]:
            self.MAX_XY_TORQUE = (self.L*self.KF*self.MAX_RPM**2)
        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)
        #### Create attributes for vision tasks ####################
        self.VISION_ATTR = vision_attributes
        if self.VISION_ATTR:
            self.IMG_RES = np.array([64, 48])
            self.IMG_FRAME_PER_SEC = 24
            self.IMG_CAPTURE_FREQ = int(self.SIM_FREQ/self.IMG_FRAME_PER_SEC)
            self.rgb = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4)))
            self.dep = np.ones(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            self.seg = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            if self.IMG_CAPTURE_FREQ%self.AGGR_PHY_STEPS != 0:
                print("[ERROR] in BaseAviary.__init__(), aggregate_phy_steps incompatible with the desired video capture frame rate ({:f}Hz)".format(self.IMG_FRAME_PER_SEC))
                exit()
            if self.RECORD:
                # TODO: This doesn't appear to work in general 
                self.ONBOARD_IMG_PATH = os.path.join(self.OUTPUT_FOLDER, "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
                os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH), exist_ok=True)
        #### Create attributes for dynamics control inputs #########
        self.DYNAMICS_ATTR = dynamics_attributes
        if self.DYNAMICS_ATTR:
            if self.DRONE_MODEL == DroneModel.CF2X:
                self.A = np.array([ [1, 1, 1, 1], [1/np.sqrt(2), 1/np.sqrt(2), -1/np.sqrt(2), -1/np.sqrt(2)], [-1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2), -1/np.sqrt(2)], [-1, 1, -1, 1] ])
            elif self.DRONE_MODEL in [DroneModel.CF2P, DroneModel.HB]:
                self.A = np.array([ [1, 1, 1, 1], [0, 1, 0, -1], [-1, 0, 1, 0], [-1, 1, -1, 1] ])
            self.INV_A = np.linalg.inv(self.A)
            self.B_COEFF = np.array([1/self.KF, 1/(self.KF*self.L), 1/(self.KF*self.L), 1/self.KM])
        #### Connect to PyBullet ###################################
        if self.GUI:
            #### With debug GUI ########################################
            self.CLIENT = p.connect(p.GUI) # p.connect(p.GUI, options="--opengl2")
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]:
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=3,
                                         cameraYaw=-30,
                                         cameraPitch=-30,
                                         cameraTargetPosition=[0, 0, 0],
                                         physicsClientId=self.CLIENT
                                         )
            ret = p.getDebugVisualizerCamera(physicsClientId=self.CLIENT)
            print("viewMatrix", ret[2])
            print("projectionMatrix", ret[3])
            if self.USER_DEBUG:
                #### Add input sliders to the GUI ##########################
                self.SLIDERS = -1*np.ones(4)
                for i in range(4):
                    self.SLIDERS[i] = p.addUserDebugParameter("Propeller "+str(i)+" RPM", 0, self.MAX_RPM, self.HOVER_RPM, physicsClientId=self.CLIENT)
                self.INPUT_SWITCH = p.addUserDebugParameter("Use GUI RPM", 9999, -1, 0, physicsClientId=self.CLIENT)
        else:
            #### Without debug GUI #####################################
            self.CLIENT = p.connect(p.DIRECT)
            #### Uncomment the following line to use EGL Render Plugin #
            #### Instead of TinyRender (CPU-based) in PYB's Direct mode
            # if platform == "linux": p.setAdditionalSearchPath(pybullet_data.getDataPath()); plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin"); print("plugin=", plugin)
            if self.RECORD:
                #### Set the camera parameters to save frames in DIRECT mode
                self.VID_WIDTH=int(640)
                self.VID_HEIGHT=int(480)
                self.FRAME_PER_SEC = 24
                self.CAPTURE_FREQ = int(self.SIM_FREQ/self.FRAME_PER_SEC)
                self.CAM_VIEW = p.computeViewMatrixFromYawPitchRoll(distance=3,
                                                                    yaw=-30,
                                                                    pitch=-30,
                                                                    roll=0,
                                                                    cameraTargetPosition=[0, 0, 0],
                                                                    upAxisIndex=2,
                                                                    physicsClientId=self.CLIENT
                                                                    )
                self.CAM_PRO = p.computeProjectionMatrixFOV(fov=60.0,
                                                            aspect=self.VID_WIDTH/self.VID_HEIGHT,
                                                            nearVal=0.1,
                                                            farVal=1000.0
                                                            )
        #### Set initial poses #####################################
        if initial_xyzs is None:
            self.INIT_XYZS = np.vstack([np.array([x*4*self.L for x in range(self.NUM_DRONES)]), \
                                        np.array([y*4*self.L for y in range(self.NUM_DRONES)]), \
                                        np.ones(self.NUM_DRONES) * (self.COLLISION_H/2-self.COLLISION_Z_OFFSET+.1)]).transpose().reshape(self.NUM_DRONES, 3)
        elif np.array(initial_xyzs).shape == (self.NUM_DRONES,3):
            self.INIT_XYZS = initial_xyzs
        else:
            print("[ERROR] invalid initial_xyzs in BaseAviary.__init__(), try initial_xyzs.reshape(NUM_DRONES,3)")
        if initial_rpys is None:
            self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        elif np.array(initial_rpys).shape == (self.NUM_DRONES, 3):
            self.INIT_RPYS = initial_rpys
        else:
            print("[ERROR] invalid initial_rpys in BaseAviary.__init__(), try initial_rpys.reshape(NUM_DRONES,3)")
        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
    
    ################################################################################

    def _dynamics(self,
                  rpm,
                  nth_drone
                  ):
        """Explicit dynamics implementation.

        Based on code written at the Dynamic Systems Lab by James Xu.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Current state #########################################
        pos = self.pos[nth_drone,:]
        quat = self.quat[nth_drone,:]
        rpy = self.rpy[nth_drone,:]
        vel = self.vel[nth_drone,:]
        rpy_rates = self.rpy_rates[nth_drone,:]
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        #### Compute forces and torques ############################
        forces = np.array(rpm**2) * self.KF
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = np.dot(rotation, thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, self.GRAVITY]) + self.WIND_FORCE
        inverse_rotation = np.array(p.getMatrixFromQuaternion([quat[0], -quat[1], -quat[2], -quat[3]])).reshape(3, 3)
        gravity_drone_frame = np.dot(inverse_rotation, np.array([0, 0, -self.GRAVITY]))
        
        # Torques from the rotors
        z_torques = np.array(rpm**2)*self.KM
        z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
        if self.DRONE_MODEL==DroneModel.CF2X:
            x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self.L/np.sqrt(2))
            y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (self.L/np.sqrt(2))
        elif self.DRONE_MODEL==DroneModel.CF2P or self.DRONE_MODEL==DroneModel.HB:
            # x_torque = (forces[1] - forces[3]) * self.L
            # y_torque = (-forces[0] + forces[2]) * self.L
            raise NotImplementedError
        torques = np.array([x_torque, y_torque, z_torque])
        # Torques from the gravity appplied to asymmetric COM
        torques = torques + np.cross(self.COM_XYZ, gravity_drone_frame)
        torques = torques - np.cross(rpy_rates, np.dot(self.J, rpy_rates))
        rpy_rates_deriv = np.dot(self.J_INV, torques)
        no_pybullet_dyn_accs = force_world_frame / self.M
        #### Update state ##########################################
        vel = vel + self.TIMESTEP * no_pybullet_dyn_accs
        rpy_rates = rpy_rates + self.TIMESTEP * rpy_rates_deriv
        pos = pos + self.TIMESTEP * vel
        rpy = rpy + self.TIMESTEP * rpy_rates
        #### Set PyBullet's state ##################################
        p.resetBasePositionAndOrientation(self.DRONE_IDS[nth_drone],
                                          pos,
                                          p.getQuaternionFromEuler(rpy),
                                          physicsClientId=self.CLIENT
                                          )
        #### Note: the base's velocity only stored and not used ####
        p.resetBaseVelocity(self.DRONE_IDS[nth_drone],
                            vel,
                            # [-1, -1, -1], # ang_vel not computed by DYN
                            rpy_rates,
                            physicsClientId=self.CLIENT
                            )
        #### Store the roll, pitch, yaw rates for the next step ####
        self.rpy_rates[nth_drone,:] = rpy_rates


    def _getDroneStateVector(self,
                             nth_drone
                             ):
        """Returns the state vector of the n-th drone.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        Returns
        -------
        ndarray 
            (20,)-shaped array of floats containing the state vector of the n-th drone.
            Check the only line in this method and `_updateAndStoreKinematicInformation()`
            to understand its format.

        """
        state = np.hstack([self.pos[nth_drone, :], self.quat[nth_drone, :], self.rpy[nth_drone, :],
                           self.vel[nth_drone, :], self.rpy_rates[nth_drone, :], self.last_clipped_action[nth_drone, :]])
        return state.reshape(20,)

    def _parseURDFParameters(self):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF)).getroot()
        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IXY = float(URDF_TREE[1][0][2].attrib['ixy'])
        IXZ = float(URDF_TREE[1][0][2].attrib['ixz'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IYZ = float(URDF_TREE[1][0][2].attrib['iyz'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.array([  [IXX, IXY, IXZ],
                        [IXY, IYY, IYZ],
                        [IXZ, IYZ, IZZ]])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        COM_XYZ = np.array([float(x) for x in URDF_TREE[10][0][0].attrib['xyz'].split(' ')])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
               GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3, COM_XYZ