import math


class Evaluate:
    '''def __int__(self,k,pre_info): #defaultdict(list)
        self.k=k
        self.pre_info
        result=self.dealt(self.pre_info)
        return result'''

    def __int__(self,k,pre_info): #defaultdict(list)
        self.k=k
        self.pre_info=pre_info


    def dealt(self,k,pre_info):   #defaultdict(list) of all users
        def takekey1(elem):
            return elem[1]
        def takekey2(elem):
            return elem[2]
        ndgc_set=[]
        hit_users=0
        users_num=len(pre_info)
        for i in range(users_num):
            real_rank=[]
            pre_rank=[]
            one_user_pre=pre_info[i]
            one_user_pre.sort(key=takekey1,reverse=True)
            for j in range(len(one_user_pre)):
                pre_rank.append(one_user_pre[j][0])  #pre_item_list
            one_user_real=pre_info[i]
            one_user_real.sort(key=takekey2,reverse=True)
            for m in range(len(one_user_real)):
                real_rank.append(one_user_real[m][0])  #real_item_list

            ndgc_set.append(self.cal_ndgc_at_k_for_each_user(k,real_rank,pre_rank))
            if self.cal_hr_at_k_for_the_user(k,real_rank,pre_rank)==1:
                hit_users+=1
        hr=hit_users/users_num
        add_ndgc=0
        for n in range(len(ndgc_set)):
            add_ndgc+=ndgc_set[n]
        ndgc=add_ndgc/users_num
        return ndgc,hr

    def cal_hr_at_k_for_the_user(self,k,real_rank,pre_rank):
        hit_user=0
        if len(real_rank)>k:real_rank=real_rank[:k]
        for i in pre_rank:
            if i in real_rank:
                hit_user=1
                break
        return hit_user

    def cal_ndgc_at_k_for_each_user(self,k,real_rank,pre_rank):  #_item_list
        idcg_k=0
        dcg_k=0
        count_num=0
        if len(real_rank)<k:
            k=len(real_rank)
        else:
            real_rank=real_rank[:k]
        for i in range(k):
            idcg_k+=1/math.log(i+2,2)
        s=set(real_rank)
        hits=[idx for idx,val in enumerate(pre_rank) if val in s]
        count=len(hits)
        for i in range(count):
            tem=hits[i]
            if tem>k:
                break
            dcg_k+=1/math.log(tem+2,2)
            count_num+=1
        '''for i in range(count_num):
            idcg_k += 1 / math.log(i + 2, 2)'''
        if idcg_k==0:
            return 0
        return float(dcg_k/idcg_k)

'''if __name__=="__main__":
    k=5
    pre_info={0:[(0,10,10),(21,2,9),(31,3,8),(41,0,7),(49,0,6),(9,9,0),(5,8,0),(6,7,0),(7,6,0),(50,5,0),(8,4,0),(1,1,0)]}
    real_rank=[0,21,31,41,49]
    pre_rank=[0,9,5,6,7,50,8,31,21,1]
    aa=Evaluate().dealt(k, pre_info)'''


