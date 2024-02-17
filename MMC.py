from random import uniform
from math import log, inf


class MMCQueue:
    """A class to represent an M/M/C queue

    Attributes
    ----------
    total_server_utilization : float
        The proportion of time that servers are busy
    num_servers : int
        Number of servers
    time_past : float
        The time between current sim time and the last event
    q_limit : int
        Maximum number of calls in the queue
    mean_arrive : float
        Mean number of calls arriving per second (lambda)
    mean_service : float
        Mean amount of time to serve call (mu)
    num_cust : int
        The number of calls that have been served
    cust_req : int
        Number of calls to serve before ending sim
    server_status : int
        The status of each server (0: Idle, 1: Busy)
    num_in_q : int
        Number of calls currently in the queue
    num_events : int
        The total number of events possible
    time_arrival : float
        The arrival times of each call
    sim_time : float
        Current simulation time
    time_last_event : float
        The time of the last event
    time_next_event : float
        The time of the next event
    next_event_type : int
        The type of the next event
    area_server_status : float
        The sum of the product of time_past and server_status, for each server

    Methods
    -------
    main():
        Runs the decision loop for the entire simulation
    timing():
        Controls the timing of the simulation
    update_time_avg_stats():
        Updates the time average of Server Status
    arrive():
        Signifies an arrival in the system
    depart():
        Signifies a departure from a given server
    report():
        Calculates the total server utilization
    expon(mean):
        Generates a number from an exponential distribution with a given mean
    block_prob():
        Calculates the blocking probability for a given number of servers
    """
    def __init__(self, q_limit: int, mean_arrive: float, mean_service: float, cust_req: int, num_servers: int):
        """Constructs all necessary attributes for M/M/C/C Queue

        Parameters
        ----------
        q_limit : int
            Maximum number of calls in the queue
        mean_arrive : float
            Mean number of calls arriving per seconds (lambda)
        mean_service : float
            Mean amount of time to serve call (mu)
        cust_req : int
            Number of calls to serve before ending sim
        num_servers : int
            Number of servers
        """
        self.total_server_utilization = 0

        self.num_servers = num_servers
        self.time_past = 0
        self.q_limit = q_limit
        self.mean_arrive = mean_arrive
        self.mean_service = mean_service
        self.num_cust = 0
        self.cust_req = cust_req
        self.server_status = [0] * num_servers
        self.num_in_q = 0
        self.num_events = num_servers + 1

        self.time_arrival = [0] * (q_limit + 1)
        self.sim_time = 0.0
        self.time_last_event = 0.0
        self.time_next_event = [inf] * (num_servers + 1)
        self.next_event_type = 0
        self.area_server_status = [0] * num_servers

    def main(self):
        """Runs the decision loop for the entire simulation

        This will run the subsequent methods in the order, until
        the required number of calls have been served:
        Timing -> Time Avg Update -> Arrive/Depart

        Returns
        -------
        None
        """
        self.time_next_event[0] = self.sim_time + self.expon(self.mean_arrive)  # Set next call arrival
        while self.num_cust < self.cust_req:
            self.timing()  # Run timing routine e.g. determine next event, set sim-time
            self.update_time_avg_stats()  # Update server status time average

            if self.next_event_type == 0:  # Next event is call arrival
                self.arrive()
            else:  # Next event is server departure
                self.depart(self.next_event_type)

        self.report()  # Calculate end of sim statistics

    def timing(self):
        """Controls the timing of the simulation

        This will determine when the next event is,
        what type of event it is and handles the change
        in simulation time.

        Returns
        -------
        None
        """
        min_time_next_event = inf - 1
        for i in range(0, self.num_events):  # For every possible event
            if self.time_next_event[i] <= min_time_next_event:  # Check if event time is less than current time
                min_time_next_event = self.time_next_event[i]  # Set current minimum to that event
                self.next_event_type = i  # Set next event type

        # Iterate sim: time of last event becomes current sim-time
        # Current sim-time becomes time of next event
        self.time_last_event = self.sim_time
        self.sim_time = self.time_next_event[self.next_event_type]

    def update_time_avg_stats(self):
        """Updates the time average of Server Status

        Returns
        -------
        None
        """
        self.time_past = self.sim_time - self.time_last_event  # Time diff between sim-time and time of last event
        for i in range(0, self.num_events-1):  # For each server, calculate time average of server status
            self.area_server_status[i] += self.time_past * self.server_status[i]

    def arrive(self):
        """Signifies an arrival in the system

        When a call arrives, it is assigned to a server
        if there is a server that is idle. If not, the
        call joins the queue if there is space. Otherwise,
        the call is considered to be blocked.

        Returns
        -------
        None
        """
        self.time_next_event[0] = self.sim_time + self.expon(self.mean_arrive)  # Schedule call arrival

        server_idle = -1
        for i in range(0, self.num_servers):  # For each server
            if self.server_status[i] == 0:  # Check if idle
                server_idle = i  # Set to first idle server and break
                break

        if server_idle != -1:  # If there is at least one idle server
            self.server_status[server_idle] = 1  # Accept call arrival
            self.time_next_event[server_idle] = self.sim_time + self.expon(self.mean_service)
            self.num_cust += 1
        elif self.num_in_q < self.q_limit:  # Add call to queue
            self.num_in_q += 1
            self.time_arrival[self.num_in_q] = self.sim_time
        # Else: call is blocked, don't need to do anything

    def depart(self, server: int):
        """Signifies a departure from a given server

        If the queue is empty, the current server is
        set as idle. Otherwise, the server takes on
        the next call in the queue.

        Parameters
        ----------
        server : int
            The server from which a departure occurs

        Returns
        -------
        None
        """
        if self.num_in_q == 0:  # If queue is empty
            self.server_status[server] = 0  # Set server status to idle
            self.time_next_event[server] = inf  # Set time of next departure for server to inf
        else:
            self.num_in_q -= 1  # Take one call from queue
            self.time_next_event[server] = self.sim_time + self.expon(self.mean_service)  # Schedule service for event

            for i in range(0, self.num_in_q + 1):  # Add arrival time to array
                self.time_arrival[i] = self.time_arrival[i + 1]

    def report(self):
        """Calculates the total server utilization

        Returns
        -------
        None
        """
        total_server_utilization = 0
        for i in range(0, self.num_servers):  # Use time average of server status to calculate server utilization
            total_server_utilization += self.area_server_status[i]

        # total_util / sim_time = time averaged utilization for all servers
        # above / num_servers = time averaged utilization per server
        self.total_server_utilization = total_server_utilization / self.sim_time / self.num_servers

    def expon(self, mean: float) -> float:
        """Generates a number from an exponential distribution with a given mean

        Parameters
        ----------
        mean : float
            The mean of the exponential distribution

        Returns
        -------
        float
            A float from an exponential distribution
        """
        return -mean * log(uniform(0, 1))

    def block_prob(self, E: float, m: int) -> float:
        """The blocking probability for a given number of servers

        Uses the Erlang B formula, running a recursive calculation.

        Parameters
        ----------
        E : float
            Offered load i.e. mean_arrive / mean_service
        m : int
            Number of Servers

        Returns
        -------
        float
            A float representing the blocking probability
        """
        inv_b = 1.0
        for j in range(1, m+1):
            inv_b = 1.0 + inv_b * j / E
        return 1.0 / inv_b


def find_max_block_prob():
    """Finds the maximum value for arrival rate

    Finds the max arrival rate value such that blocking
    probability is < 0.01

    Returns
    -------
    max_value : float
        Float for the maximum arrival rate value
    """
    mean_arrival = 1  # Mean starts at 1 to avoid float rep errors
    last_mean = 0.0
    done = False

    while not done:
        q = MMCQueue(
            q_limit=0,
            mean_arrive=mean_arrival/100,
            mean_service=100,
            cust_req=100,
            num_servers=16,
        )
        E = q.mean_arrive / q.mean_service
        blocking_prob = q.block_prob(E, q.num_servers)

        if blocking_prob >= 0.01:  # Blocking probability starts very low and increases
            # So check that blocking prob is >= to 0.01 to find maximum arrival rate to get under 0.01
            done = True
        else:
            last_mean = mean_arrival
            mean_arrival += 1

    return last_mean/100  # Divide int arrival to provide the original float


if __name__ == '__main__':
    # Create values for graph
    util_arr = []
    block_arr = []

    for i in range(1, 11, 1):
        q = MMCQueue(
            q_limit=0,
            mean_arrive=i/100,
            mean_service=100,
            cust_req=100,
            num_servers=16,
        )
        q.main()
        util_arr.append(q.total_server_utilization)
        E = q.mean_arrive / q.mean_service
        block_arr.append(q.block_prob(E, q.num_servers))

    print(util_arr)
    print(block_arr)

    # Find maximum arrival rate for BP < 0.01
    max_arrival = find_max_block_prob()
    print(max_arrival)

    q = MMCQueue(
        q_limit=0,
        mean_arrive=max_arrival,
        mean_service=100,
        cust_req=100,
        num_servers=16,
    )
    q.main()
    E = q.mean_arrive / q.mean_service
    print(q.block_prob(E, q.num_servers))
