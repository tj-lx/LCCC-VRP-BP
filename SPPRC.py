import heapq
from functools import cmp_to_key
import numpy as np
from route import Route
from paramsVRP import ParamsVRP
from sortedcontainers import SortedSet
from functools import total_ordering

'''
 shortest path with resource constraints
 inspired by Irnish and Desaulniers, "SHORTEST PATH PROBLEMS WITH RESOURCE CONSTRAINTS"
 for educational demonstration only - (nearly) no code optimization

 four main lists will be used:
 labels: array (ArraList) => one dimensional unbounded vector
        list of all labels created along the feasible paths (i.e. paths satisfying the resource constraints)

 U: sorted list (TreeSet) => one dimensional unbounded vector
        sorted list containing the indices of the unprocessed labels (paths that can be extended to obtain a longer feasible path)

 P: sorted list (TreeSet) => one dimensional unbounded vector
        sorted list containing the indices of the processed labels ending at the depot with a negative cost

 city2labels: matrix (array of ArrayList) => nbclients x unbounded
        for each city, the list of (indices of the) labels attached to this city/vertex
        before processing a label at vertex i, we compare pairwise all labels at the same vertex to remove the dominated ones
'''


class SPPRC:

    def __init__(self, userParam=None):
        self.paramsVRP = ParamsVRP() if userParam is None else userParam
        self.labels = []
        #self.myLabelComparator = self.MyLabelComparator(self)

    @total_ordering
    class label:
        def __init__(self, city, index_prev_label, cost, ttime, demand, dominated, vertex_visited, parent):
            self.city = city  # int
            self.index_prev_label = index_prev_label  # int
            self.cost = cost  # double
            self.ttime = ttime  # float
            self.demand = demand  # double
            self.dominated = dominated  # boolean
            self.vertex_visited = vertex_visited  # boolean list
            self.parent = parent

        def updateLabel(self, a1, a2, a3, a4, a5, a6, a7):
            self.city = a1  # current vertex
            self.index_prev_label = a2  # previous label in the same path
            # (i.e. previous vertex in the same path with the state of the resources)
            self.cost = a3  # first resource: cost (e.g. distance or strict travel time)
            self.ttime = a4  # second resource: travel time along the path (including wait time and service time)
            self.demand = a5  # third resource: demand,i.e. total quantity delivered to the clients encountered on this path
            self.dominated = a6  # is this label dominated by another one? i.e. if dominated, forget this path.
            self.vertex_visited = a7

        def __lt__(self, other):
            if self.cost - other.cost < -1e-7:
                return True
            elif self.cost - other.cost > 1e-7:
                return False
            else:
                if self.city == other.city:
                    if self.ttime - other.ttime < -1e-7:
                        return True
                    elif self.ttime - other.ttime > 1e-7:
                        return False
                    else:
                        if self.demand - other.demand < -1e-7:
                            return True
                        elif self.demand - other.demand > 1e-7:
                            return False
                        else:
                            i = 0
                            while i < self.parent.paramsVRP.nbclients:
                                if A.vertex_visited[i] != B.vertex_visited[i]:
                                    if A.vertex_visited[i]:
                                        return True
                                    else:
                                        return False
                                i += 1
                            return False
                elif self.city > other.city:
                    return False
                else:
                    return True


        def __eq__(self, other):
            if self.cost - other.cost < -1e-7 or self.cost - other.cost > 1e-7:
                return False
            if self.city < other.city or self.city > other.city:
                return False
            if self.ttime - other.ttime < -1e-7 or self.ttime - other.ttime > 1e-7:
                return False
            if self.demand - other.demand < -1e-7 or self.demand - other.demand > 1e-7:
                return False

            if self.cost - other.cost > -1e-7 and self.cost - other.cost < 1e-7:
                if self.city == other.city:
                    if self.ttime - other.ttime > -1e-7 and self.ttime - other.ttime < 1e-7:
                        if self.demand - other.demand > -1e-7 and self.demand - other.demand < 1e-7:
                            i = 0
                            while i < self.parent.paramsVRP.nbclients:
                                if self.vertex_visited[i] != other.vertex_visited[i]:
                                    if self.vertex_visited[i]:
                                        return False
                                    else:
                                        return False
                                i += 1
                            return True

    def shortestPath(self, userParamArg, routes, nbroute):
        print("[---SPPRC.shortestPath called---]")
        self.paramsVRP = userParamArg

        # Initialize unprocessed labels list (U) and processed labels list (P)
        U = SortedSet(key=lambda x: x)
        P = SortedSet(key=lambda x: x)

        # Initialize labels array        labels = []
        # for depot 0
        cust = [False] * (self.paramsVRP.nbclients)
        cust[0] = True
        self.labels.append(self.label(0, -1, 0.0, 0, 0, False, cust, self))  # First label: start from depot (client 0)
        U.add(0)

        # For each city, an array with the index of the corresponding labels (for dominance)
        checkDom = [0] * self.paramsVRP.nbclients # 每个客户节点 被检查过“占优性”的节点有多少个
        city2labels = [[] for _ in range(self.paramsVRP.nbclients)]
        city2labels[0].append(0)
        #print("checkDom", checkDom)
        #print("city2labels:", city2labels)

        nbsol = 0
        maxsol = 2 * nbroute
        iteration_count = 0
        MAX_ITERATIONS = 20000  # Safety break
        MAX_LABELS_PER_NODE = 30 # Heuristic pruning

        while U and nbsol < maxsol and iteration_count < MAX_ITERATIONS:
            iteration_count += 1
            #print("U:", U)
            current_idx = 0
            current_idx = U.pop(0)  # Process one label => get the index AND remove it from U
            current = self.labels[current_idx]
            
            # Pruning: If too many labels at this node, remove worst ones
            # This is a heuristic!
            current_node_labels = city2labels[current.city]
            if len(current_node_labels) > MAX_LABELS_PER_NODE:
                # Sort by cost
                current_node_labels.sort(key=lambda idx: self.labels[idx].cost)
                # Keep best ones
                kept_labels = current_node_labels[:MAX_LABELS_PER_NODE]
                removed_labels = current_node_labels[MAX_LABELS_PER_NODE:]
                
                city2labels[current.city] = kept_labels
                
                # Mark removed as dominated so they don't expand
                for ridx in removed_labels:
                    self.labels[ridx].dominated = True
                    if ridx in U:
                        U.discard(ridx)
                
                # If current label was removed, skip expansion
                if current.dominated:
                    continue

            # Check for dominance
            cleaning = []
            for i in range(checkDom[current.city], len(city2labels[current.city])):
                for j in range(i):
                    l1, l2 = city2labels[current.city][i], city2labels[current.city][j]
                    la1, la2 = self.labels[l1], self.labels[l2]
                    if not la1.dominated and not la2.dominated and l1 != l2:

                        # Q1：判断 标签2 是否被占优
                        pathdom = True
                        for k in range(1, self.paramsVRP.nbclients):
                            if not pathdom:
                                break
                            pathdom = pathdom and (not la1.vertex_visited[k] or la2.vertex_visited[k])
                        if pathdom and la1.cost <= la2.cost and la1.ttime <= la2.ttime and la1.demand <= la2.demand:
                            #print(f'U:{U}')
                            #print(f'[l2({l2}).dominated=True]labels:{[label.dominated for label in self.labels]}')
                            U.discard(l2)
                            self.labels[l2].dominated = True
                            cleaning.append(l2)
                            pathdom = False

                        pathdom = True
                        # Q2：判断 标签1 是否被占优
                        for k in range(1, self.paramsVRP.nbclients):
                            pathdom = pathdom and (not la2.vertex_visited[k] or la1.vertex_visited[k])
                        if pathdom and la2.cost <= la1.cost and la2.ttime <= la1.ttime and la2.demand <= la1.demand:
                            #print(f'U:{U}')
                            #print(f'[l1({l1}).dominated=True]labels:{[label.dominated for label in self.labels]}')
                            U.discard(l1)
                            self.labels[l1].dominated = True
                            cleaning.append(l1)
                            j = len(city2labels[current.city])

            for c in cleaning:
                city2labels[current.city].remove(c)
            cleaning = None

            # 更新CheckDom：所有在city2labels的label都检查过dominance
            checkDom[current.city] = len(city2labels[current.city])
            #print(f'U:{U}, checkDom:{checkDom}')

            # Expand REF
            if not current.dominated:
                #print(f'[current_idx]:{current_idx} is not dominated')
                if current.city == self.paramsVRP.nbclients - 1:  # Shortest path candidate to the depot!
                    if current.cost < -1e-7:  # SP candidate for the column generation
                        P.add(current_idx)
                        #print(f'[current_idx ADDED]:{current_idx}')
                        nbsol = sum(1 for labi in P if not self.labels[labi].dominated)
                else:  # If not the depot, we can consider extensions of the path
                    for i in range(self.paramsVRP.nbclients):
                        if not current.vertex_visited[i] and self.paramsVRP.dist[current.city][i] < self.paramsVRP.verybig - 1e-6:
                            # Calculate Departure time from current city
                            start_service_time = max(current.ttime, self.paramsVRP.a[current.city])
                            departure_time = start_service_time + self.paramsVRP.s[current.city]
                            
                            tt = departure_time + self.paramsVRP.ttime[current.city][i]
                            
                            # Hard Time Window Check (Depot Closing Time & Max Lateness)
                            # Assume depot (0) closing time is the global limit.
                            global_limit = self.paramsVRP.b[0]
                            
                            # Node-specific hard limit (Soft TW with Max Lateness)
                            # If max_lateness is 0, it becomes Hard TW.
                            node_limit = self.paramsVRP.b[i] + self.paramsVRP.max_lateness
                            
                            d = current.demand + self.paramsVRP.d[i]

                            if tt <= global_limit and tt <= node_limit and d <= self.paramsVRP.capacity:
                                # Calculate Dynamic Costs
                                dynamic_cost = 0.0
                                # If i is a customer (not depot start or end)
                                if i != 0 and i != self.paramsVRP.nbclients - 1:
                                     # Freshness: P_fresh * q_i * theta * t_i
                                     freshness_cost = self.paramsVRP.P_fresh * self.paramsVRP.d[i] * self.paramsVRP.theta * tt
                                     
                                     # Soft TW Penalty: alpha * max(0, tt - b[i])
                                     tw_penalty = self.paramsVRP.alpha * max(0, tt - self.paramsVRP.b[i])
                                     
                                     # Door Cost
                                     door_cost = self.paramsVRP.c_door
                                     
                                     dynamic_cost = freshness_cost + tw_penalty + door_cost
                                
                                new_cost = current.cost + self.paramsVRP.cost[current.city][i] + dynamic_cost

                                idx = len(self.labels)
                                newcust = current.vertex_visited[:]
                                newcust[i] = True

                                # Feillet's optimization (unreachable nodes)
                                for j in range(1, self.paramsVRP.nbclients - 1):
                                    if not newcust[j]:
                                        # Estimate earliest arrival at j from i
                                        start_service_at_i = max(tt, self.paramsVRP.a[i])
                                        dep_i = start_service_at_i + self.paramsVRP.s[i]
                                        tt2 = dep_i + self.paramsVRP.ttime[i][j]
                                        d2 = d + self.paramsVRP.d[j]
                                        
                                        # Check against global limit
                                        if tt2 > global_limit or d2 > self.paramsVRP.capacity:
                                            newcust[j] = True

                                self.labels.append(self.label(i, current_idx, new_cost, tt, d, False, newcust, self))
                                if idx not in U:
                                    U.add(idx)
                                    city2labels[i].append(idx)
                                else:
                                    self.labels[idx].dominated = True

        # Filtering: find the path from depot to the destination
        i = 0
        checkDom = None
        while i < nbroute and P:
            lab = P.pop(0)
            if not self.labels[lab].dominated:
                if self.labels[lab].cost < -1e-4:
                    route = Route()
                    route.set_cost(self.labels[lab].cost)
                    route.add_city(self.labels[lab].city)
                    path = self.labels[lab].index_prev_label
                    while path >= 0:
                        route.add_city(self.labels[path].city)
                        path = self.labels[path].index_prev_label
                    route.switch_path()
                    routes.append(route)
                    #print(f'[route ADDED]:{route}', f'[route.cost]:{route.get_cost()}', f'[route.path]:{route.get_path()}')
                    i += 1

