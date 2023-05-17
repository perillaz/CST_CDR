import numpy as np
import torch
from collections import defaultdict

class Clustering:

    def __init__(self,center_user,user_list,max_k,common_user_list):
        self.center_user = center_user
        self.user_list=user_list
        self.max_k=max_k
        self.common_user_list=common_user_list
        #self.max_iter=max_iter
        self.cluster_list = defaultdict(set)


    def ckmeans(self):
        for i in self.center_user:
            temp_k=[[]]
            k=self.max_k
            if k==self.max_k:
                for j in self.user_list[i]:
                    self.cluster_list[i].add(j)
                    temp_k[self.max_k-k].append(j)
                    self.common_user_list[j]=1
                k=k-1
            else:
                while k>0:
                    for j in temp_k[self.max_k-k-1]:
                        self.cluster_list[i].update(self.user_list[j])
                        temp_k[self.max_k-k].extend(list(self.user_list[j]))
                        self.common_user_list[j] = 1
                    k=k-1
            self.cluster_list[i].add(i)
        return self.cluster_list









