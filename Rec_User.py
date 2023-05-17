import torch

class Rec_User:
    def __init__(self,enc_u,center_users,cluster_list,common_users,common_user_list):
        self.enc_u=enc_u
        self.center_users=center_users
        self.cluster_list=cluster_list
        self.common_users=common_users
        self.common_user_list=common_user_list

    def best_circle(self):
        for i in self.common_users:
            best=0
            max_similar=0
            #i是新用户或者边缘用户，没有加入到聚类中
            if self.common_user_list[i]==0:
                for j in self.center_users:
                    temp_similar=torch.cosine_similarity(self.enc_u[i], self.enc_u[j])
                    if torch.gt(temp_similar,max_similar):
                        max_similar=temp_similar
                        best=j
                self.cluster_list[best].add(i)
                self.common_user_list[i]=1





