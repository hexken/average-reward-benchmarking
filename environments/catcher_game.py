import numpy as np


class CatcherGame:
    """Catcher
    Agent is a 3 pixel `basket` in the bottom of a square grid of size
    `grid_size`. Single pixels `fruits` fall from the top and return +1 reward
    if catch and -1 else.
    Attributes
    ----------
    grid_size : int
        Size of the square grid
    output_type: str
        Either give state description as raw 'pixels', or as the location of
        the fruit and basket 'position'. The 'pixels' state space size is
        2**(grid_size**2), while the 'position' state space size is
        grid_size**3.
    """

    def __init__(self, grid_size=10, output_shape=None, output_type='position'):
        self.grid_size = grid_size
        self.output_type = output_type
        self.output_shape = output_shape

        if output_shape is None:
            if output_type == 'pixels':
                output_shape = (grid_size ** 2,)
            elif output_type == 'position':
                output_shape = (3,)
        self.output_shape = output_shape
        self.reset()

    def _update_state(self, action):
        """
        Input: action and states
        Ouput: new states and reward
        """
        state = self.state
        if action == 0:  # left
            action = -1
        elif action == 1:  # stay
            action = 0
        else:
            action = 1  # right
        f0, f1, basket = state[0]
        new_basket = min(max(1, basket + action), self.grid_size - 1)
        f0 += 1
        out = np.asarray([f0, f1, new_basket])
        out = out[np.newaxis]

        assert len(out.shape) == 2
        self._state = out

    def _draw_state(self):
        """Convert state description into a square image
        """
        im_size = (self.grid_size,) * 2
        state = self.state[0]
        canvas = np.zeros(im_size)
        canvas[state[0], state[1]] = 1  # draw fruit
        canvas[-1, state[2] - 1:state[2] + 2] = 1  # draw basket
        return canvas

    def reward(self):
        fruit_row, fruit_col, basket = self.state[0]
        if fruit_row == self.grid_size - 1:
            if abs(fruit_col - basket) <= 1:
                return 1
            else:
                return -1
        else:
            return 0

    def observe_image(self):
        canvas = self._draw_state()
        return canvas.reshape(1, 1, self.grid_size, self.grid_size)

    def observe(self):
        if self.output_type == 'pixels':
            canvas = self._draw_state()
            out = canvas.reshape((1,) + self.output_shape)
        if self.output_type == 'position':
            out = self.state[0]
        return out

    def update(self, action):
        self._update_state(action)
        reward = self.reward()
        #TODO make game continuing
        game_over = False
        return reward, self.observe(), game_over

    def reset(self):
        n = np.random.randint(0, self.grid_size - 1, size=1)
        m = np.random.randint(1, self.grid_size - 2, size=1)
        self._state = np.asarray([0, n, m])[np.newaxis]

    @property
    def state(self):
        return self._state

    @property
    def is_over(self):
        if self.state[0, 0] == self.grid_size - 1:
            return True
        else:
            return False

    @property
    def description(self):
        return f"Catch game with grid size {self.grid_size}"
