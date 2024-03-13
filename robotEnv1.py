__credits__ = ["Yuri De Santis"]

from typing import Optional

# NUMPY & GYMNASIUM
import numpy        as np
import gymnasium    as gym
from gymnasium      import spaces

# PINOCCHIO & EXAMPLE_ROBOT_DATA
import pinocchio as pin
from pinocchio.robot_wrapper    import RobotWrapper as robWrap
from pinocchio.visualize        import MeshcatVisualizer
from example_robot_data         import load

# STABLE_BASELINES3
from stable_baselines3          import PPO
#from stable_baselines3.common.env_util import make_vec_env

# ADDITIONAL
import subprocess

ROB_STR = 'double_pendulum'
DT = 0.05

class PinDoublePendulumEnv(gym.Env):

    def __init__(self, rob_str:str, render_bool:bool = False, dt:float=0.05):
        """
        Args:
            rob_str (str): robot string name of example_robot_data (pip install example_robot_data)
            render_bool (Optional[str], optional): Defaults to None.
            g (float, optional): gravity used in dynamic. Defaults to 10.0.
        """
        robot:robWrap = load(rob_str)
        self.frame_rate = 1/30
        # pinocchio objects
        self.robModel       = robot.model               # urdf info
        self.robData        = robot.data                # useful functions
        self.robColl        = robot.collision_model
        self.robVis         = robot.visual_model
        
        # get joints dim
        self.dim_q:int      = self.robModel.nq
        self.dim_dq:int     = self.robModel.nv
        
        # get joint position ranges
        robM = self.robModel
        if isinstance(robM.upperPositionLimit, np.ndarray) and isinstance(robM.lowerPositionLimit, np.ndarray):
            if robM.upperPositionLimit.all() == 0 and robM.lowerPositionLimit.all() == 0:
                Warning("upperPositionLimit and lowerPositionLimit are set as zeros as default. \n \
                    To have a range, the limit is set to np.finfo(np.float32).max")
                q_max = np.ones(robM.upperPositionLimit.shape)*np.finfo(np.float32).max
                q_min = -q_max
            else:
                q_min = robM.lowerPositionLimit
                q_max = robM.upperPositionLimit
        else:
            raise("Error robot.model.lowerPositionLimit and robot.model.upperPositionLimit is not np.ndarray")
        # get velocity ranges
        if isinstance(robM.velocityLimit, np.ndarray):
            if robM.velocityLimit.all() == 0:
                Warning("Velocity are set as zeros as default. \n \
                    To have a range the limit is set to np.finfo(np.float32).max")
                dq_max = np.ones(robM.velocityLimit.shape)*np.finfo(np.float32).max
            else:
                dq_max = robM.velocityLimit
        else:
            raise("Error robot.model.velocityLimit is not np.ndarray")
        # get effort ranges
        if isinstance(robM.effortLimit, np.ndarray):
            if robM.effortLimit.all() == 0:
                Warning("Effort are set as zeros as default. \n \
                    To have a range, the limit is set to np.finfo(np.float32).max")
                u_max = np.ones(robM.effortLimit.shape)*np.finfo(np.float32).max
            else:
                u_max = robM.effortLimit
        else:
            raise("Error robot.model.effortLimit is not np.ndarray")
        # simmetric ranges
        dq_min  = -dq_max
        u_min   = -u_max
        
        # concatenate limits for action
        self.lowLimA    = u_min
        self.highLimA   = u_max
        # concatenate limits for observation
        self.lowLimO    = np.concatenate((q_min, dq_min, u_min), axis=0)
        self.highLimO   = np.concatenate((q_min, dq_min, u_max), axis=0)
        # concatenate limits for state
        self.lowLimq    = q_min
        self.highLimq   = q_max
        self.lowLimDq   = dq_min
        self.highLimDq  = dq_max
        self.lowLimS    = np.concatenate((q_min, dq_min), axis=0)
        self.highLimS   = np.concatenate((q_min, dq_min), axis=0)
        
        # action.space
        self.action_space       = spaces.Box(low=self.lowLimA, high=self.highLimA, shape=self.lowLimA.shape, dtype=np.float32)
        # observation.space
        self.observation_space  = spaces.Box(low=self.lowLimO, high=self.highLimO, shape=self.lowLimO.shape, dtype=np.float32)

        # init pinocchio object and update all functions
        q0:np.ndarray   = pin.neutral(self.robModel)
        dq0:np.ndarray  = np.zeros(q0.shape)
        pin.computeAllTerms(self.robModel,self.robData,q0,dq0)
        
        self.q      = q0
        self.dq     = dq0
        self.u      = np.zeros(u_min.shape)
        self.state  = self._get_state()
        self.obs    = self._get_obs()

        self.dt = dt
        self.render_bool = render_bool
        self.isopen = False
        
        # start visualization
        if self.render_bool:
            
            self.viz = MeshcatVisualizer(self.robModel, self.robColl, self.robVis)
            self.viz.initViewer(open=True)            
            self.viz.loadViewerModel("pinocchio")
            self.viz.display(self.q)
            meshcat_url = self.viz.viewer.url()
            subprocess.run(['open', meshcat_url], check=True)    
            self.isopen=True
                   
    def step(self, u:np.ndarray):
        
        u_clip = np.clip(u, self.lowLimA, self.highLimA)
        
        dt          = self.dt
        robModel    = self.robModel
        robData     = self.robData

        # convert array in vector
        u_clip      = u_clip.reshape(-1,1)
        q           = self.q.reshape(-1,1)
        dq          = self.dq.reshape(-1,1)
        
        pin.computeAllTerms(robModel,robData,q,dq)
        #pin.computeCoriolisMatrix(robModel,robData,q,dq)
        #pin.computeGeneralizedGravity(robModel,robData,q)
        Minvmat:np.ndarray  = pin.computeMinverse(robModel,robData,q)
        Cmat:np.ndarray     = robData.C
        Gmat:np.ndarray     = robData.g.reshape(-1,1)
        
        ddq:np.ndarray  = np.matmul(Minvmat,np.matmul(-Cmat,dq)-Gmat-u_clip)
        
        # implementare runge_kutta
        dq_new          = dq + ddq*dt
        q_new           = angle_normalize(q + 1/2*dq_new*dt)
        
        q_clip  = np.clip(q_new.flatten(), self.lowLimq, self.highLimq)
        dq_clip = np.clip(dq_new.flatten(), self.lowLimDq, self.highLimDq)
        self.q  = q_clip
        self.dq = dq_clip
        self.u  = u_clip.flatten()
        
        reward = np.sum(angle_normalize(q_clip) ** 2) + 0.1 * np.sum(dq_clip**2) + 0.001 * np.sum((u_clip**2))

        self.state          = self._get_state()
        self.obs            = self._get_obs()
        
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(), -reward, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options is None:
            self.q      = self.np_random.uniform(low=self.lowLimq, high=self.highLimq)
            self.dq     = self.np_random.uniform(low=self.lowLimDq, high=self.highLimDq)*0    
            self.u      = np.zeros(self.u.shape)
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            """ x = options.get("x_init") if "x_init" in options else DEFAULT_X
            y = options.get("y_init") if "y_init" in options else DEFAULT_Y
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            high = np.array([x, y]) """
            1

        
        pin.computeAllTerms(self.robModel,self.robData,self.q,self.dq)
    
        self.state  = self._get_state()
        self.obs    = self._get_obs()
        
        return self.obs, {}

    def _get_obs(self) -> np.ndarray:
        state = self._get_state()
        u = self.u
        obs = np.concatenate([state, u], axis=0, dtype=np.float32)
        return obs

    def _get_state(self) -> np.ndarray:
        state = np.concatenate([self.q, self.dq], axis=0, dtype=np.float32)
        return state
    
    def render(self):
        if not self.render_bool:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return
        else:
            self.viz.display(self.q)
            self.viz.sleep(self.frame_rate)
        
    def close(self):
        if self.render_bool is not None:
            self.viz.delete()
            self.isopen = False

def angle_normalize(x:np.ndarray):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


env = PinDoublePendulumEnv(rob_str=ROB_STR, render_bool=True)
# env.reset()
# u= np.array(np.random.rand(2,))
# env.step(u)
# a = env._get_obs()
# Parallel environments
#vec_env = make_vec_env("CartPole-v1", n_envs=4)

# model = PPO("MlpPolicy", env)
# model.learn(total_timesteps=10)
# model.save("ppo_cartpole")

# del model # remove to demonstrate saving and loading
# model = PPO.load("ppo_cartpole")

obs, _ = env.reset()
print("passato", obs)
# time.sleep(10)

k = 0
while True:
    #action, _states = model.predict(obs)
    action = 0*np.array(np.random.rand(2))
    obs, rewards, dones, _, info = env.step(action)
    k +=1
    env.render()