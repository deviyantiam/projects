import pandas as pd
xx={'Recency':[16,50,143],'Frequency':[1,3,5],'Monetary':[293.36,648.08,1611.72]}
xx=pd.DataFrame(xx)
class RFM:
    def __init__(self):
        # self.input_id=None
        pass

    def RScore(x):
        rs=0
        for i in range(3):
            if x <= xx.iloc[i,0]:
                rs+=1
                break
            elif x > xx.iloc[2,0]:
                rs=4
                break
        return rs

    def FScore(x):
        rs=4
        for i in range(3):
            if x <= xx.iloc[i,1]:
                rs-=1
                break
            elif x > xx.iloc[2,1]:
                rs=1
                break
        return rs

    def MScore(x):
        rs=4
        for i in range(3):
            if x <= xx.iloc[i,2]:
                rs-=1
                break
            elif x > xx.iloc[2,2]:
                rs=1
                break
        return rs

class Fin_RFM:
    def __init__(self):
        # self.input_id=None
        pass

    def XScore(x):
        xs=0
        if x == '111':
            xs=0
        elif x in ('112','113','114','212','213','214','312','313','314','412','413','414'):
            xs=1
        elif x in ('121','131','141','211','221','231','241','321','331','341','421','431','441'):
            xs=2
        elif x == '331':
            xs=3
        elif x == '411':
            xs=4
        elif x == '444':
            xs=5
        else:
            xs=6
        cust_label={0:'Best Customers',1:'Loyal Customers',2:'Big Spenders',3:'Almost Lost',4:'Lost Customers',5:'Lost Cheap',6:'Others'}
        ans_rfm=cust_label[xs]
        return ans_rfm
