from random import uniform
from math import log, inf


class M1M2MCQueue:
    """A class to represent an M1+M2/M/C/C Queue

    Attributes
    ----------
    handover_call_arrivals : int
        The total number of handover calls that arrive
    new_call_arrivals : int
        The total number of new calls that arrive
    new_calls_lost : int
        The total number of new calls that are blocked
    handover_calls_lost : int
        The total number of handover calls that are blocked
    total_server_utilization : float
        The proportion of time that servers are busy
    num_servers : int
        Number of servers
    time_past : float
        The time between current sim time and the last event
    q_limit : int
        Maximum number of calls in the queue
    mean_m1_arrive : float
        Mean number of new calls arriving per second (lambda 1)
    mean_m2_arrive : float
        Mean number of handover calls arriving per second (lambda 2)
    mean_service : float
        Mean amount of time to serve call (mu)
    threshold : int
        Number of idle servers required to accept new calls
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
    arrive(type):
        Signifies an arrival (either new or handover) in the system
    depart():
        Signifies a departure from a given server
    report():
        Calculates the total server utilization
    expon(mean):
        Generates a number from an exponential distribution with a given mean
    agg_block_prob():
        Calculates the aggregated blocking probability
    total_idle():
        Finds the total number of currently idle servers
    """
    def __init__(self, q_limit: int, mean_m1_arrive: float, mean_m2_arrive: float, mean_service: float, cust_req: int,
                 num_servers: int, threshold: int):
        """Constructs all necessary attributes for M1+M2/M/C/C Queue

        Parameters
        ----------
        q_limit : int
            Maximum number of calls in the queue
        mean_m1_arrive : float
            Mean number of new calls arriving per second (lambda 1)
        mean_m2_arrive : float
            Mean number of handover calls arriving per second (lambda 2)
        mean_service : float
            Mean amount of time to serve call (mu)
        cust_req : int
            Number of calls to serve before ending sim
        num_servers : int
            Number of servers
        threshold : int
            Number of idle servers required to accept new calls
        """
        self.handover_call_arrivals = 0
        self.new_call_arrivals = 0
        self.new_calls_lost = 0
        self.handover_calls_lost = 0
        self.total_server_utilization = 0

        self.num_servers = num_servers
        self.time_past = 0
        self.q_limit = q_limit
        self.mean_m1_arrive = mean_m1_arrive  # New Call Arrival Rate
        self.mean_m2_arrive = mean_m2_arrive  # Handover Call Arrival Rate
        self.mean_service = mean_service
        self.threshold = threshold  # Number of idle servers to accept

        self.num_cust = 0
        self.cust_req = cust_req  # Number of customers required to end sim
        self.server_status = [0] * num_servers  # Status of each server
        self.num_in_q = 0
        self.num_events = num_servers + 2  # Events are departure from all 16 servers and 2 arrival types
        self.time_arrival = [0] * (q_limit + 1)
        self.sim_time = 0.0
        self.time_last_event = 0.0
        self.time_next_event = [inf] * (num_servers + 2)  # Need space for 16 departure events and 2 arrival events
        self.next_event_type = 0
        self.area_server_status = [0] * num_servers

    def main(self):
        """Runs the decision loop for the entire simulation

        This will run the subsequent methods in order until
        the required number of calls has been served:
        Timing -> Time Avg Update -> Arrive/Depart

        Returns
        -------
        None
        """
        self.time_next_event[0] = self.sim_time + self.expon(self.mean_m1_arrive)  # Set next new call arrival
        self.time_next_event[1] = self.sim_time + self.expon(self.mean_m2_arrive)  # Set next handover call arrival
        while self.num_cust < self.cust_req:
            self.timing()  # Run timing routine e.g. determining next event, setting sim-time
            self.update_time_avg_stats()  # Update the server status time average

            if self.next_event_type == 0:  # Next event is new call arrival
                self.new_call_arrivals += 1
                self.arrive(type=1)
            elif self.next_event_type == 1:  # Next event is handover call arrival
                self.handover_call_arrivals += 1
                self.arrive(type=2)
            else:  # Next event is a server departure
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
            if self.time_next_event[i] <= min_time_next_event:  # Check if event time is less than current closest event
                min_time_next_event = self.time_next_event[i]  # Set current minimum to that event
                self.next_event_type = i  # Set the next event type

        # Iterate simulation: time of last event becomes current sim-time and
        # current sim-time becomes time of next event
        self.time_last_event = self.sim_time
        self.sim_time = self.time_next_event[self.next_event_type]

    def update_time_avg_stats(self):
        """Updates the time average of Server Status

        Returns
        -------
        None
        """
        self.time_past = self.sim_time - self.time_last_event  # Time difference between sim-time and time of last event
        for i in range(0, self.num_events-2):  # For each server, calculate the time average of server status
            self.area_server_status[i] += self.time_past * self.server_status[i]

    def arrive(self, type: int):
        """Signifies an arrival in the system

        If a new call arrives, it is assigned to the
        first idle server if there are more than the
        threshold required e.g. more than 2 idle servers.
        If a handover call arrives, it is assigned to the
        first idle server. If there are no idle servers,
        the call is blocked.

        Parameters
        ----------
        type : int
            The type of arrival where 1 = new call, 2 = handover call

        Returns
        -------
        None
        """
        if type == 2:
            self.time_next_event[1] = self.sim_time + self.expon(self.mean_m2_arrive)  # Schedule handover call
        elif type == 1:
            self.time_next_event[0] = self.sim_time + self.expon(self.mean_m1_arrive)  # Schedule new call

        server_idle = -1
        for i in range(0, self.num_servers):  # For each server
            if self.server_status[i] == 0:  # Check if idle
                server_idle = i  # Set first idle server and break
                break

        total_free = self.total_idle()  # Total number of currently idle servers

        if server_idle != -1:  # If there is at least one server idle
            if type == 2 and total_free > 0:  # Accept priority handover call arrival
                self.server_status[server_idle] = 1
                self.time_next_event[server_idle+2] = self.sim_time + self.expon(self.mean_service)
                self.num_cust += 1
            elif type == 1 and total_free > self.threshold:  # Accept a new call arrival
                self.server_status[server_idle] = 1
                self.time_next_event[server_idle+2] = self.sim_time + self.expon(self.mean_service)
                self.num_cust += 1
            else:  # call is blocked due to not reaching idle threshold
                if type == 1:
                    self.new_calls_lost += 1
                elif type == 2:
                    self.handover_calls_lost += 1
        elif self.num_in_q < self.q_limit:  # Add call to queue
            self.num_in_q += 1
            self.time_arrival[self.num_in_q] = self.sim_time
        else:  # call is blocked due to not enough idle servers
            if type == 1:
                self.new_calls_lost += 1
            elif type == 2:
                self.handover_calls_lost += 1

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
            self.server_status[server-2] = 0  # Set server status to idle
            self.time_next_event[server] = inf  # Set time of departure for that server to inf
        else:  # If queue has elements
            self.num_in_q -= 1  # Take one call from queue
            self.time_next_event[server-2] = self.sim_time + self.expon(self.mean_service)  # Schedule service for event

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

        # Calculate server utilization
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

    def agg_block_prob(self) -> float:
        """Calculates the aggregated blocking probability

        Returns
        -------
        float
            Float representing aggregated blocking probability
        """
        try:
            call_block_prob = self.new_calls_lost / self.new_call_arrivals
        except ZeroDivisionError:  # If number of arrivals is zero, set call blocking prob to zero
            call_block_prob = 0
        try:
            handover_fail_prob = self.handover_calls_lost / self.handover_call_arrivals
        except ZeroDivisionError:  # If number of arrivals is zero, set handover fail prob to zero
            handover_fail_prob = 0

        agg_block_prob = call_block_prob + (10 * handover_fail_prob)  # Calculate AGB
        print(f"ABP: {agg_block_prob}")
        return agg_block_prob

    def total_idle(self) -> int:
        """Finds the total number of currently idle servers

        Returns
        -------
        int
            An int representing the number of idle servers
        """
        total_idle = 0
        for i in range(0, self.num_servers):
            if self.server_status[i] == 0:
                total_idle += 1

        return total_idle


def find_max_handover_arrival() -> float:
    """Finds the maximum handover arrival rate such that ABP < 0.02.

    Returns
    -------
    float
        The maximum handover arrival rate such that ABP < 0.02
    """
    mean_arrival = 1  # Mean starts at 1 to avoid float rep errors
    last_mean = 0.0
    done = False

    while not done:
        q = M1M2MCQueue(  # Instantiate new queue
            q_limit=0,
            mean_m1_arrive=0.1,
            mean_m2_arrive=mean_arrival/10,
            mean_service=100,
            cust_req=100,
            num_servers=16,
            threshold=2,
        )
        q.main()  # Run simulation
        print(f"Mean Arrival: {mean_arrival/10}")
        abp = q.agg_block_prob()  # Calculate ABP for sim

        if abp < 0.02:  # If the blocking prob is less than 0.02, stop searching
            done = True
        else:
            last_mean = mean_arrival
            mean_arrival += 1

    return last_mean/10  # Divide int arrival to provide the original float


def find_max_new_arrival() -> float:
    """Finds the maximum new call arrival rate such that ABP < 0.02.

    Returns
    -------
    float
        The maximum new call arrival rate such that ABP < 0.02
    """
    mean_arrival = 1  # Mean starts at 1 to avoid float rep errors
    last_mean = 0.0
    done = False

    while not done:
        q = M1M2MCQueue(  # Instantiate new queue
            q_limit=0,
            mean_m1_arrive=mean_arrival/10,
            mean_m2_arrive=0.03,
            mean_service=100,
            cust_req=100,
            num_servers=16,
            threshold=2,
        )
        q.main()  # Run simulation
        print(f"Mean Arrival: {mean_arrival/10}")
        abp = q.agg_block_prob()  # Calculate ABP for sim

        if abp < 0.02:  # If the blocking prob is less than 0.02, stop searching
            done = True
        else:
            last_mean = mean_arrival
            mean_arrival += 1

    return last_mean/10  # Divide int arrival to provide the original float


if __name__ == '__main__':
    print(find_max_handover_arrival())
    print(find_max_new_arrival())
