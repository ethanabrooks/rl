import cv2


class SpaceWrapper:
    def __init__(self, shape, scale=1.):
        self.shape = list(shape)
        for dim in range(2):
            self.shape[dim] *= scale


def preprocess(image, scale):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(image, (0, 0), fx=scale, fy=scale)


class EnvWrapper:
    def __init__(self, env, scale=1.):
        self.__dict__.update(env.__dict__)
        self._env = env
        self._scale = scale
        self.observation_space = SpaceWrapper(env.observation_space.shape[:2], scale)

    def reset(self):
        return preprocess(self._env.reset(), self._scale)

    def step(self, action):
        observation, reward, done, info = self._env.step(action)
        return preprocess(observation, self._scale), reward, done, info
