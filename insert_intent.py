import pandas as pd
import numpy as np

if __name__ == "__main__": 
        deep_Kmean = "kmean_three_level.csv"
        df = pd.read_csv("comsci_result/"+deep_Kmean) # get data from deep Kmean

        label_group = df.centroids_id.unique() # get label number
        df["intent"] = np.NaN
        df['target'] = np.NaN

        intent_name = ['ทักทาย', 'ข้อมูลหลักสูตร', 'ระยะเวลาเรียน', 'ค่าเทอม', 'เรียนแมธ', 'ข้อมูลหอพัก', 'งานในอนาคต', 'ความยากในการเรียน', 'วันเปิดเทอม', 'วัดปิดเทอม', 'วันสอบกลางภาค', 'วันสอบปลายภาค', 'สายที่รับสมัคร', 'ที่ตั้งสาขา', 'การสมัครเรียน', 'เว็บไซต์มหาลัย', 'เว็บไซต์คณะ', 'เว็บไซต์สาขา', 'วันลงทะเบียนเรียน', 'วิธีการดรอปเรียน', 'ข้อมูลติดต่อสาขา', 'ขอบคุณ', 'วันสอบทั้งหมด' ]

        # for i in range(len(intent_name)):
        #         print (i , ' : ', intent_name[i])
        count_target = -1
        for i in label_group: # insert label name by user
                insert_label = df[df.centroids_id == i]
                ex_text = insert_label[insert_label.dist_score == insert_label.dist_score.min()]\
                                                        .text.sample(1).to_string(index=False)
                print(i,") example message : ", ex_text)
                target_ = int(input('number intent : '))
                # label_name = input("assigned intent : ")
                # intent = df.intent.tolist()
                # target = -1
                # if label_name not in intent:
                #         count_target += 1
                #         target = count_target
                # else:
                #         target = df[df.intent == label_name].target.tolist()[0]
                print('-----------------------------')
                # df.at[df.centroids_id == i, "intent"] = label_name
                # df.at[df.centroids_id == i, "target"] = target
                df.at[df.centroids_id == i, "intent"] = intent_name[target_]
                df.at[df.centroids_id == i, "target"] = target_
        
        label_name = df.intent.unique().tolist()

        data_ = pd.DataFrame({'target':df.target, 'intent':df.intent, 'text':df.text })
        save_at = "data_training/intent_group.csv"
        data_.to_csv(save_at,encoding='utf-8-sig',index=False)







        