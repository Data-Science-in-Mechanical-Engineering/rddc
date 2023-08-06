import numpy as np

class SimpleStateFeedbackController():
    """
    """

    ################################################################################

    def __init__(self, n=12, m=12, K=None):
        """
        n - number of system states
        m - number of control variables, which are reference states
        """
        if K is None:
            #K = np.array([[]])
            K = np.zeros((m,n))
        
        assert(K.shape[0] == m)
        assert(K.shape[1] == n)
        self.K = K

    def computeControl( self,
                        error_pos,
                        error_vel,
                        error_rpy,
                        error_rpy_rates
                        ):
    
        new_error = self.K @ np.hstack([[error_pos], [error_vel], [error_rpy], [error_rpy_rates]]).T

        return new_error.T[0]
