from gym_multigrid.multigrid import *
from ipdb import set_trace
from parameters import GRID, MID_WALL, TOTAL_AGENTS


# ------ TwoRooms
class OneRoomGameEnv(MultiGridEnv):
    """
    Environment in which the agents have to fetch the balls and drop them in their respective goals
    """

    def __init__(
        self,
        size=10,
        view_size=3,
        width=None,
        height=None,
        goal_pst = [],
        goal_index = [],
        num_balls=[],
        agents_index = [],
        balls_index=[],
        zero_sum = False,

    ):

        self.num_balls = num_balls
        self.goal_pst = goal_pst
        self.goal_index = goal_index
        self.balls_index = balls_index
        self.zero_sum = zero_sum

        self.one_hot = self.get_one_hot(GRID*GRID)
        self.world = World

        self.view_size = view_size

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps= 10000,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=view_size
        )

    def _gen_grid(self, width, height):

        # create grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        # # Create a vertical splitting wall
        # # == Top
        # self.grid.vert_wall(self.world, int(width / 2), 1, int(height/3)-2)
        #
        # # == Bottom
        # self.grid.vert_wall(self.world, int(width/2), int(height/3))
        #
        # # == Right Horizontal Left
        # self.grid.horz_wall(self.world, int(height / 2), int(width / 2), int(width/3)-1)
        #
        # # == Right Horizontal Right
        # self.grid.horz_wall(self.world, int(height / 2)+int(height/2) - 3, int(width / 2), int(width/3)-1)
        #
        # # == Left Horizontal Left
        # self.grid.horz_wall(self.world, 0, int(height / 2), int(width / 2) - 4)
        #
        # # == Left Horizontal Right
        # self.grid.horz_wall(self.world, int(height / 2)-3, int(height / 2), int(width / 2) - 4)

        # ---- Place goal
        loc_x, loc_y = width-2, height-2
        self.goal_state = self.tile_state[(loc_y, loc_x)]
        self.atomic_reward[self.goal_state] = 1

        # loc_x, loc_y = int(width/4), int(height/2)
        # loc_x, loc_y = width-2, height-2
        # loc_x, loc_y = 1, int(height/2) + 1


        self.place_obj(Goal(self.world, 1, 'goal'), top=[loc_x,loc_y], size=[1,1])

        # Randomize the player start position and orientation
        for a in self.agents:
            self._place_agent(a, top=(1, int(height/2) + 1))

    def _place_agent(
            self,
            agent,
            top=None,
            size=None,
            rand_dir=True,
            max_tries=math.inf
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """

        agent.pos = None

        pos = self.place_obj(agent, top, size, max_tries=max_tries)
        agent.pos = pos
        agent.init_pos = pos

        if rand_dir:
            agent.dir = self._rand_int(0, 4)

        agent.init_dir = agent.dir

        return pos

    def _reward(self, i, rewards,reward=1):

        # print("%%%%%%%%%%")
        # print(rewards)


        for j,a in enumerate(self.agents):
            # if a.index==i or a.index==0:
            # --- James
            if a.index == i:
                rewards[j]+=reward



            if self.zero_sum:
                if a.index!=i or a.index==0:
                    rewards[j] -= reward

        # print(rewards)
        # print("#############")

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                if self.agents[i].carrying is None:
                    self.agents[i].carrying = fwd_cell
                    self.agents[i].carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
            elif fwd_cell.type=='agent':
                if fwd_cell.carrying:
                    if self.agents[i].carrying is None:
                        self.agents[i].carrying = fwd_cell.carrying
                        fwd_cell.carrying = None

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        if self.agents[i].carrying:
            if fwd_cell:
                if fwd_cell.type == 'objgoal' and fwd_cell.target_type == self.agents[i].carrying.type:
                    if self.agents[i].carrying.index in [0, fwd_cell.index]:
                        self._reward(fwd_cell.index, rewards, fwd_cell.reward)
                        self.agents[i].carrying = None
                elif fwd_cell.type=='agent':
                    if fwd_cell.carrying is None:
                        fwd_cell.carrying = self.agents[i].carrying
                        self.agents[i].carrying = None
            else:
                self.grid.set(*fwd_pos, self.agents[i].carrying)
                self.agents[i].carrying.cur_pos = fwd_pos
                self.agents[i].carrying = None

    def step(self, actions):

        obs, rewards, done, info, nt_new, ntsa, loc, ntsas, goal_flag   = MultiGridEnv.step(self, actions)
        assert nt_new.sum() <= TOTAL_AGENTS
        # ---- New Reward
        new_rew = self.atomic_reward * self.nt_new
        new_rew = new_rew.sum()
        # set_trace()
        return obs, new_rew, done, info, nt_new, ntsa, loc, ntsas, goal_flag

    def get_one_hot(self, n_classes):
        target = tc.tensor([[_ for _ in range(n_classes)]])
        y = tc.zeros(n_classes, n_classes).type(tc.int)
        y[range(y.shape[0]), target] = 1
        y = y.data.numpy()
        return y

class OneRoomEnvNxN(OneRoomGameEnv):
    def __init__(self):

        ag_ind = []
        for i in range(TOTAL_AGENTS):
            ag_ind.append(i)


        super().__init__(size=None,
        height=GRID,
        width=GRID,
        goal_index=[1],
        agents_index = ag_ind,
        zero_sum=False)

# ------ TwoRooms
class TwoRoomGameEnv(MultiGridEnv):
    """
    Environment in which the agents have to fetch the balls and drop them in their respective goals
    """

    def __init__(
        self,
        size=10,
        view_size=3,
        width=None,
        height=None,
        goal_pst = [],
        goal_index = [],
        num_balls=[],
        agents_index = [],
        balls_index=[],
        zero_sum = False,

    ):

        self.num_balls = num_balls
        self.goal_pst = goal_pst
        self.goal_index = goal_index
        self.balls_index = balls_index
        self.zero_sum = zero_sum

        self.one_hot = self.get_one_hot(GRID*GRID)
        self.world = World

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps= 10000,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=view_size
        )

    def _gen_grid(self, width, height):

        # create grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        # Create a vertical splitting wall
        if MID_WALL:
            # == Top
            self.grid.vert_wall(self.world, int(width / 2), 1, int(width/2)-1)

            # == Bottom
            self.grid.vert_wall(self.world, int(width/2), int(height/2)+1)

        # ---- Place goal
        loc_x, loc_y = width-2, height-2
        # loc_x, loc_y = width - 7, height - 7
        # print(loc_x, loc_y)
        # exit()

        # loc_x, loc_y = 1, 3

        self.place_obj(Goal(self.world, 1, 'goal'), top=[loc_x,loc_y], size=[1,1])

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)

    def _reward(self, i, rewards,reward=1):

        # print("%%%%%%%%%%")
        # print(rewards)

        for j,a in enumerate(self.agents):
            # if a.index==i or a.index==0:
            # --- James
            if a.index == i:
                rewards[j]+=reward
            if self.zero_sum:
                if a.index!=i or a.index==0:
                    rewards[j] -= reward

        # print(rewards)
        # print("#############")

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                if self.agents[i].carrying is None:
                    self.agents[i].carrying = fwd_cell
                    self.agents[i].carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
            elif fwd_cell.type=='agent':
                if fwd_cell.carrying:
                    if self.agents[i].carrying is None:
                        self.agents[i].carrying = fwd_cell.carrying
                        fwd_cell.carrying = None

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        if self.agents[i].carrying:
            if fwd_cell:
                if fwd_cell.type == 'objgoal' and fwd_cell.target_type == self.agents[i].carrying.type:
                    if self.agents[i].carrying.index in [0, fwd_cell.index]:
                        self._reward(fwd_cell.index, rewards, fwd_cell.reward)
                        self.agents[i].carrying = None
                elif fwd_cell.type=='agent':
                    if fwd_cell.carrying is None:
                        fwd_cell.carrying = self.agents[i].carrying
                        self.agents[i].carrying = None
            else:
                self.grid.set(*fwd_pos, self.agents[i].carrying)
                self.agents[i].carrying.cur_pos = fwd_pos
                self.agents[i].carrying = None

    def step(self, actions):
        obs, rewards, done, info, nt_new, ntsa, loc, ntsas, goal_flag   = MultiGridEnv.step(self, actions)

        nb_agents = len(actions)
        nt_new2 = nt_new.reshape(1, GRID*GRID)
        ns_new_tmp = np.tile(nt_new2, (nb_agents, 1))

        # s = loc.reshape(nb_agents, 1)
        s_new_tmp = self.one_hot[loc]

        s_nt_new = np.concatenate((s_new_tmp, ns_new_tmp), axis=1)

        return obs, rewards, done, info, nt_new, ntsa, loc, s_nt_new, ntsas, goal_flag

    def get_one_hot(self, n_classes):
        target = tc.tensor([[_ for _ in range(n_classes)]])
        y = tc.zeros(n_classes, n_classes).type(tc.int)
        y[range(y.shape[0]), target] = 1
        y = y.data.numpy()
        return y

class TwoRoomEnv10x10(TwoRoomGameEnv):
    def __init__(self):
        super().__init__(size=None,
        height=GRID,
        width=GRID,
        goal_index=[1],
        agents_index = [0, 1],
        zero_sum=False)

# ------ ThreeRooms
class ThreeRoomGameEnv(MultiGridEnv):
    """
    Environment in which the agents have to fetch the balls and drop them in their respective goals
    """

    def __init__(
        self,
        size=10,
        view_size=3,
        width=None,
        height=None,
        goal_pst = [],
        goal_index = [],
        num_balls=[],
        agents_index = [],
        balls_index=[],
        zero_sum = False,

    ):

        self.num_balls = num_balls
        self.goal_pst = goal_pst
        self.goal_index = goal_index
        self.balls_index = balls_index
        self.zero_sum = zero_sum

        self.one_hot = self.get_one_hot(GRID*GRID)
        self.world = World

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps= 10000,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=view_size
        )

    def _gen_grid(self, width, height):

        # create grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        # Create a vertical splitting wall
        if MID_WALL:
            # == Top
            self.grid.vert_wall(self.world, int(width / 2), 1, int(width/3)-1)

            # == Bottom
            self.grid.vert_wall(self.world, int(width/2), int(height/3)+1)

            # == Horizontal Left
            self.grid.horz_wall(self.world, int(height / 2), int(width / 2)+1, int(width/3)-1)

            # == Horizontal Right
            self.grid.horz_wall(self.world, int(height / 2)+int(height/2) - 3, int(width / 2)+1, int(width/3)-1)


        # ---- Place goal
        loc_x, loc_y = width-2, height-2
        # loc_x, loc_y = 1, 3

        self.place_obj(Goal(self.world, 1, 'goal'), top=[loc_x,loc_y], size=[1,1])

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)

    def _reward(self, i, rewards,reward=1):

        # print("%%%%%%%%%%")
        # print(rewards)

        for j,a in enumerate(self.agents):
            # if a.index==i or a.index==0:
            # --- James
            if a.index == i:
                rewards[j]+=reward
            if self.zero_sum:
                if a.index!=i or a.index==0:
                    rewards[j] -= reward

        # print(rewards)
        # print("#############")

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                if self.agents[i].carrying is None:
                    self.agents[i].carrying = fwd_cell
                    self.agents[i].carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
            elif fwd_cell.type=='agent':
                if fwd_cell.carrying:
                    if self.agents[i].carrying is None:
                        self.agents[i].carrying = fwd_cell.carrying
                        fwd_cell.carrying = None

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        if self.agents[i].carrying:
            if fwd_cell:
                if fwd_cell.type == 'objgoal' and fwd_cell.target_type == self.agents[i].carrying.type:
                    if self.agents[i].carrying.index in [0, fwd_cell.index]:
                        self._reward(fwd_cell.index, rewards, fwd_cell.reward)
                        self.agents[i].carrying = None
                elif fwd_cell.type=='agent':
                    if fwd_cell.carrying is None:
                        fwd_cell.carrying = self.agents[i].carrying
                        self.agents[i].carrying = None
            else:
                self.grid.set(*fwd_pos, self.agents[i].carrying)
                self.agents[i].carrying.cur_pos = fwd_pos
                self.agents[i].carrying = None

    def step(self, actions):
        obs, rewards, done, info, nt_new, ntsa, loc, ntsas, goal_flag   = MultiGridEnv.step(self, actions)

        nb_agents = len(actions)
        nt_new2 = nt_new.reshape(1, GRID*GRID)
        ns_new_tmp = np.tile(nt_new2, (nb_agents, 1))

        # s = loc.reshape(nb_agents, 1)
        s_new_tmp = self.one_hot[loc]
        s_nt_new = np.concatenate((s_new_tmp, ns_new_tmp), axis=1)

        return obs, rewards, done, info, nt_new, ntsa, loc, s_nt_new, ntsas, goal_flag

    def get_one_hot(self, n_classes):
        target = tc.tensor([[_ for _ in range(n_classes)]])
        y = tc.zeros(n_classes, n_classes).type(tc.int)
        y[range(y.shape[0]), target] = 1
        y = y.data.numpy()
        return y

class ThreeRoomEnvNxN(ThreeRoomGameEnv):
    def __init__(self):
        super().__init__(size=None,
        height=GRID,
        width=GRID,
        goal_index=[1],
        agents_index = [0, 1],
        zero_sum=False)

# ------ FourRooms
class FourRoomGameEnv(MultiGridEnv):
    """
    Environment in which the agents have to fetch the balls and drop them in their respective goals
    """

    def __init__(
        self,
        size=10,
        view_size=3,
        width=None,
        height=None,
        goal_pst = [],
        goal_index = [],
        num_balls=[],
        agents_index = [],
        balls_index=[],
        zero_sum = False,

    ):

        self.num_balls = num_balls
        self.goal_pst = goal_pst
        self.goal_index = goal_index
        self.balls_index = balls_index
        self.zero_sum = zero_sum

        self.one_hot = self.get_one_hot(GRID*GRID)
        self.world = World

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps= 10000,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=view_size
        )

    def _gen_grid(self, width, height):

        # create grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        # Create a vertical splitting wall
        # == Top
        self.grid.vert_wall(self.world, int(width / 2), 1, int(height/3)-2)

        # == Bottom
        self.grid.vert_wall(self.world, int(width/2), int(height/3))

        # == Right Horizontal Left
        self.grid.horz_wall(self.world, int(height / 2), int(width / 2), int(width/3)-1)

        # == Right Horizontal Right
        self.grid.horz_wall(self.world, int(height / 2)+int(height/2) - 3, int(width / 2), int(width/3)-1)

        # == Left Horizontal Left
        self.grid.horz_wall(self.world, 0, int(height / 2), int(width / 2) - 4)

        # == Left Horizontal Right
        self.grid.horz_wall(self.world, int(height / 2)-3, int(height / 2), int(width / 2) - 4)

        # ---- Place goal
        loc_x, loc_y = int(width/2)+ 4, int(height/2)
        # loc_x, loc_y = int(width/4), int(height/2)

        # loc_x, loc_y = width-2, height-2
        # loc_x, loc_y = 1, int(height/2) + 1


        self.place_obj(Goal(self.world, 1, 'goal'), top=[loc_x,loc_y], size=[1,1])

        # Randomize the player start position and orientation
        for a in self.agents:
            self._place_agent(a, top=(1, int(height/2) + 1))

    def _place_agent(
            self,
            agent,
            top=None,
            size=None,
            rand_dir=True,
            max_tries=math.inf
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """

        agent.pos = None

        pos = self.place_obj(agent, top, size, max_tries=max_tries)
        agent.pos = pos
        agent.init_pos = pos

        if rand_dir:
            agent.dir = self._rand_int(0, 4)

        agent.init_dir = agent.dir

        return pos

    def _reward(self, i, rewards,reward=1):

        # print("%%%%%%%%%%")
        # print(rewards)

        for j,a in enumerate(self.agents):
            # if a.index==i or a.index==0:
            # --- James
            if a.index == i:
                rewards[j]+=reward
            if self.zero_sum:
                if a.index!=i or a.index==0:
                    rewards[j] -= reward

        # print(rewards)
        # print("#############")

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                if self.agents[i].carrying is None:
                    self.agents[i].carrying = fwd_cell
                    self.agents[i].carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
            elif fwd_cell.type=='agent':
                if fwd_cell.carrying:
                    if self.agents[i].carrying is None:
                        self.agents[i].carrying = fwd_cell.carrying
                        fwd_cell.carrying = None

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        if self.agents[i].carrying:
            if fwd_cell:
                if fwd_cell.type == 'objgoal' and fwd_cell.target_type == self.agents[i].carrying.type:
                    if self.agents[i].carrying.index in [0, fwd_cell.index]:
                        self._reward(fwd_cell.index, rewards, fwd_cell.reward)
                        self.agents[i].carrying = None
                elif fwd_cell.type=='agent':
                    if fwd_cell.carrying is None:
                        fwd_cell.carrying = self.agents[i].carrying
                        self.agents[i].carrying = None
            else:
                self.grid.set(*fwd_pos, self.agents[i].carrying)
                self.agents[i].carrying.cur_pos = fwd_pos
                self.agents[i].carrying = None

    def step(self, actions):
        obs, rewards, done, info, nt_new, ntsa, loc, ntsas, goal_flag   = MultiGridEnv.step(self, actions)

        nb_agents = len(actions)
        nt_new2 = nt_new.reshape(1, GRID*GRID)
        ns_new_tmp = np.tile(nt_new2, (nb_agents, 1))

        # s = loc.reshape(nb_agents, 1)
        s_new_tmp = self.one_hot[loc]
        s_nt_new = np.concatenate((s_new_tmp, ns_new_tmp), axis=1)

        return obs, rewards, done, info, nt_new, ntsa, loc, s_nt_new, ntsas, goal_flag

    def get_one_hot(self, n_classes):
        target = tc.tensor([[_ for _ in range(n_classes)]])
        y = tc.zeros(n_classes, n_classes).type(tc.int)
        y[range(y.shape[0]), target] = 1
        y = y.data.numpy()
        return y

class FourRoomEnvNxN(FourRoomGameEnv):
    def __init__(self):
        super().__init__(size=None,
        height=GRID,
        width=GRID,
        goal_index=[1],
        agents_index = [0, 1],
        zero_sum=False)
