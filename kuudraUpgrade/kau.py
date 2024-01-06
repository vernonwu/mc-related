import pandas as pd
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

class kau:
    def __init__(self,start_tier,end_tier):
        '''
        Initializes the kau class with the specified start and end tiers.

        Parameters:
        - start_tier (int): The starting tier for the algorithm.
        - end_tier (int): The ending tier for the algorithm.
        '''
        self.start_tier = start_tier
        self.end_tier = end_tier
        self.base_mean = 1
        self.base_var = 0.5
        self.mean_multiplier = 2
        self.var_multiplier = 2
        self.uuid_map = {}
        self.data = pd.DataFrame()
        self.total_num = 0
        self.G = nx.DiGraph()

    def get_nums(self):
        '''
        Calculate the total required number of tier n nodes.
        '''
        nums = []
        nums.append(pow(2,self.end_tier-1)-pow(2,self.start_tier-1))

        for i in range(self.end_tier-1):
            nums_n = int(nums[-1]/2) 
            if i == self.start_tier-2:
                nums.append(nums_n+1) # original node inclusive
            else:
                nums.append(max(2,nums_n)) # 2 nodes in the last tier to avoid pd.DataFrame error

        self.nums = nums
    
    def generate_data(self):
        '''
        Generate the data for each tier using normal distribution.
        '''
        df = pd.DataFrame(columns=['UUID', 'Tier', 'Price'])
        mean, var = self.base_mean, self.base_var

        for tier in range(self.end_tier):

            # Generate data for this tier
            required_num = self.nums[tier]
            prices = []
            while len(prices) < required_num:
                new_prices = np.random.normal(mean, var, size=required_num - len(prices))
                new_prices = new_prices[new_prices > mean - var] # Remove outliers
                prices.extend(new_prices)
            print('Generated', len(prices), 'tier', tier + 1, 'items.')

            uuids = range(self.total_num + 1, self.total_num + len(prices) + 1)
            tiers = [tier + 1] * len(prices)
            tier_df = pd.DataFrame({'UUID': uuids, 'Tier': tiers, 'Price': prices})
            df = pd.concat([df, tier_df])

            mean *= self.mean_multiplier
            var *= self.var_multiplier
            self.total_num += len(prices)

        df = df.sort_values(['Tier', 'Price']).set_index('Tier')

        self.data = df
    
    def fused_node(self,current_tier,id_start) -> pd.DataFrame:
        '''
        Combine 2 nodes in the current tier to a fused node.

        Parameters:
        - current_tier (int): The current tier.
        - id_start (int): The starting index of the nodes to be fused.

        Returns:
        - pd.DataFrame: A dataframe with the fused node.
        '''
        left_elem =  self.data.loc[current_tier].iloc[id_start] # id_start serves as left ptr
        right_elem = self.data.loc[current_tier].iloc[id_start+1]

        new_uuid = self.total_num + 1
        new_price = left_elem['Price'] + right_elem['Price']
        new_tier = current_tier + 1

        self.uuid_map[new_uuid] = [left_elem['UUID'],right_elem['UUID']]
        return pd.DataFrame({'UUID': [new_uuid], 'Tier': [new_tier], 'Price': [new_price]}).set_index('Tier')

        
    def insert_fused_node(self,next_tier,fused_node):
        '''
        Insert fused node to the dataframe.

        Parameters:
        - next_tier (int): The next tier.
        - fused_node (pd.DataFrame): The fused node to be inserted.

        Returns:
        - int: The inserted position.
        '''
        for i in range(len(self.data.loc[next_tier])):
            if pd.DataFrame(self.data.loc[next_tier]).iloc[i]['Price'] >= fused_node.iloc[0]['Price']: # find the first node with higher price

                for j in range(len(self.data.loc[next_tier])-1,i,-1):
                    self.data.loc[next_tier].iloc[j] = self.data.loc[next_tier].iloc[j-1] # shift right

                self.data.loc[next_tier].iloc[i] = fused_node.iloc[0] # insert fused node
                return i
        return -1

    
    def insertion_pop(self) -> pd.DataFrame:
        '''
        Repeatedly insert fused nodes until the corresponding size is achieved.
        '''
        current_tier = 1
        while current_tier < self.end_tier:
            print('Tier',current_tier,'started...')

            # insert original node (to the leftmost of its tier)
            if current_tier == self.start_tier:
                original = pd.DataFrame([['0',current_tier,0]],columns = ['UUID','Tier','Price']).set_index('Tier')
                _ = self.insert_fused_node(current_tier,original) 

            required_num = self.nums[current_tier]
            for i in range(0,len(self.data.loc[current_tier]),2): # left ptr step = 2
                fused = self.fused_node(current_tier,i)
                inserted_id = self.insert_fused_node(current_tier+1,fused)
                # if the inserted node is the last node or the required size is achieved, break
                if inserted_id == -1 or inserted_id > required_num-1: 
                    break

                self.total_num += 1

            print('Tier',current_tier,'is completed.')
            current_tier += 1

        print('All tiers are completed.')
    
    def get_total_cost(self):
        '''
        Calculate the total cost and direct cost.
        '''
        self.total_cost = self.data.loc[self.end_tier].iloc[0]['Price']
        self.direct_cost = self.data.loc[self.end_tier].iloc[1]['Price']

    def plot_tree(self):
        '''
        Plot the tree.
        '''
        res = self.data.loc[self.end_tier].iloc[0]['UUID']
        self.G.add_node(res)
        queue = [res]
        origin = None

        # Populate the graph and labels using BFS
        while queue:
            node = queue.pop(0)
            children = self.uuid_map.get(node, [])
            for child in children:
                if child == '0':
                    origin = child
                self.G.add_edge(node, child)
                queue.append(child)

        pos = graphviz_layout(self.G, prog='dot')

        leaf_nodes = [n for n in self.G if self.G.out_degree(n) == 0 and n != '0']
        original_node = origin
        non_leaf_nodes = [n for n in self.G if self.G.out_degree(n) != 0]

        # Graph layout
        pos = graphviz_layout(self.G, prog='dot')
        self.G = nx.DiGraph.reverse(self.G)

        nx.draw_networkx_nodes(self.G, pos, nodelist=leaf_nodes, node_color='#009999', node_size=500, alpha=0.8)
        nx.draw_networkx_nodes(self.G, pos, nodelist=non_leaf_nodes, node_color='#00CC66', node_size=500, alpha=0.8)
        nx.draw_networkx_nodes(self.G, pos, nodelist=original_node, node_color='#00CCCC', node_size=500, alpha=0.8)

        nx.draw_networkx_edges(self.G, pos, width=1.0, alpha=0.5)

        nx.draw_networkx_labels(self.G, pos, labels={n: n for n in leaf_nodes} , font_size=10)

        plt.show()

        self.auctions = leaf_nodes

    def check_dependencies(self):
        '''
        Check for dependencies in environment.yml and install if necessary.
        '''
        import sys
        import subprocess
        import pkg_resources

        required = {'pandas','numpy','networkx','matplotlib'}
        installed = {pkg.key for pkg in pkg_resources.working_set}
        missing = required - installed

        if missing:
            python = sys.executable
            subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

    def run(self):
        '''
        Run the kau algorithm.
        '''
        self.check_dependencies()
        self.get_nums()
        self.generate_data()
        self.insertion_pop()
        self.get_total_cost()

        