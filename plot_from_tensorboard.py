# https://www.tensorflow.org/tensorboard/dataframe_api

from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb


# defender - 2
experiment_id_11 = '~/THESIS/report_10/csvs/run-110800-tag-cumulative_hack_probability_eval.csv'
experiment_id_12 = '~/THESIS/report_10/csvs/run-210800-tag-cumulative_hack_probability_eval.csv'
experiment_id_13 = '~/THESIS/report_10/csvs/run-310800-tag-cumulative_hack_probability_eval.csv'
experiment_id_14 = '~/THESIS/report_10/csvs/run-defensereinforce-tag-cumulative_hack_probability_eval.csv'

# attacker env 1 - 3
experiment_id_21 = '~/THESIS/report_10/csvs/run-1101111100-tag-cumulative_hack_probability_eval.csv'
experiment_id_22 = '~/THESIS/report_10/csvs/run-1108888800-tag-cumulative_hack_probability_eval.csv'
experiment_id_23 = '~/THESIS/report_10/csvs/run-1109999911-tag-cumulative_hack_probability_eval.csv'


# attacker env 2 - 4
experiment_id_31 = '~/THESIS/report_10/csvs/run-1201111100-tag-cumulative_hack_probability_eval.csv'
experiment_id_32 = '~/THESIS/report_10/csvs/run-1208888800-tag-cumulative_hack_probability_eval.csv'
experiment_id_33 = '~/THESIS/report_10/csvs/run-1209999911-tag-cumulative_hack_probability_eval.csv'
experiment_id_34 = '~/THESIS/report_10/csvs/run-1209999912-tag-cumulative_hack_probability_eval.csv'

# attacker env 0 - 5
experiment_id_41 = '~/THESIS/report_10/csvs/run-1001111100-tag-cumulative_hack_probability_eval.csv'
experiment_id_42 = '~/THESIS/report_10/csvs/run-1008888800-tag-cumulative_hack_probability_eval.csv'
experiment_id_43 = '~/THESIS/report_10/csvs/run-1009999911-tag-cumulative_hack_probability_eval.csv'
experiment_id_44 = '~/THESIS/report_10/csvs/run-1009999912-tag-cumulative_hack_probability_eval.csv'


df11 = pd.read_csv(experiment_id_11)
df12 = pd.read_csv(experiment_id_12)
df13 = pd.read_csv(experiment_id_13)
df14 = pd.read_csv(experiment_id_14)

df21 = pd.read_csv(experiment_id_21)
df22 = pd.read_csv(experiment_id_22)
df23 = pd.read_csv(experiment_id_23)

df31 = pd.read_csv(experiment_id_31)
df32 = pd.read_csv(experiment_id_32)
df33 = pd.read_csv(experiment_id_33)
df34 = pd.read_csv(experiment_id_34)

df41 = pd.read_csv(experiment_id_41)
df42 = pd.read_csv(experiment_id_42)
df43 = pd.read_csv(experiment_id_43)
df44 = pd.read_csv(experiment_id_44)


'''
df1 = pd.DataFrame({"1":df11["Value"].rolling(window=1, axis=0).mean(),
                    "2":df12["Value"].rolling(window=1, axis=0).mean(),
                   "3":df13["Value"].rolling(window=1, axis=0).mean()})

df2 = pd.DataFrame({"1":df21["Value"].rolling(window=1, axis=0).mean(),
                    "2":df22["Value"].rolling(window=1, axis=0).mean(),
                   "3":df23["Value"].rolling(window=1, axis=0).mean()})

df3 = pd.DataFrame({"1":df31["Value"].rolling(window=1, axis=0).mean(),
                    "2":df32["Value"].rolling(window=1, axis=0).mean(),
                   "3":df33["Value"].rolling(window=1, axis=0).mean()})
'''
#######################################################################################



#hexperiment_id_11 = '~/THESIS/results_external_GPU/run-21000-tag-cumulative_hack_probability_eval.csv'
#hexperiment_id_12 = '~/THESIS/results_external_GPU/run-21100-tag-cumulative_hack_probability_eval.csv'
#hexperiment_id_13 = '~/THESIS/results_external_GPU/run-21200-tag-cumulative_hack_probability_eval.csv'

#hexperiment_id_21 = '~/THESIS/results_external_GPU/run-22000-tag-cumulative_hack_probability_eval.csv'
#hexperiment_id_22 = '~/THESIS/results_external_GPU/run-22100-tag-cumulative_hack_probability_eval.csv'
#hexperiment_id_23 = '~/THESIS/results_external_GPU/run-22200-tag-cumulative_hack_probability_eval.csv'

#hexperiment_id_31 = '~/THESIS/results_external_GPU/run-23000-tag-cumulative_hack_probability_eval.csv'
#hexperiment_id_32 = '~/THESIS/results_external_GPU/run-23100-tag-cumulative_hack_probability_eval.csv'
#hexperiment_id_33 = '~/THESIS/results_external_GPU/run-23200-tag-cumulative_hack_probability_eval.csv'


#hdf11 = pd.read_csv(hexperiment_id_11)
#hdf12 = pd.read_csv(hexperiment_id_12)
#hdf13 = pd.read_csv(hexperiment_id_13)

#hdf21 = pd.read_csv(hexperiment_id_21)
#hdf22 = pd.read_csv(hexperiment_id_22)
#hdf23 = pd.read_csv(hexperiment_id_23)

#hdf31 = pd.read_csv(hexperiment_id_31)
#hdf32 = pd.read_csv(hexperiment_id_32)
#hdf33 = pd.read_csv(hexperiment_id_33)


#hdf1 = pd.DataFrame({"1":hdf11["Value"].rolling(window=10, axis=0).mean(),
#                    "2":hdf12["Value"].rolling(window=10, axis=0).mean(),
#                   "3":hdf13["Value"].rolling(window=10, axis=0).mean()})

#hdf2 = pd.DataFrame({"1":hdf21["Value"].rolling(window=10, axis=0).mean(),
#                    "2":hdf22["Value"].rolling(window=10, axis=0).mean(),
#                   "3":hdf23["Value"].rolling(window=10, axis=0).mean()})

#hdf3 = pd.DataFrame({"1":hdf31["Value"].rolling(window=10, axis=0).mean(),
#                    "2":hdf32["Value"].rolling(window=10, axis=0).mean(),
#                   "3":hdf33["Value"].rolling(window=10, axis=0).mean()})








if __name__ == '__main__':

    plt.figure(figsize=(20, 60))

    #plt.subplot(3, 1, 1)

    plt.figure()
    plt.plot(df11["Step"], df11["Value"], color='red')
    plt.plot(df12["Step"], df12["Value"], color='blue')
    plt.plot(df13["Step"], df13["Value"], color='green')
    plt.plot(df14["Step"], df14["Value"][:5001], color='orange', )
    #plt.fill_between(df11["Step"], df1.min(axis=1), df1.max(axis=1), 1)
    #plt.title('Attacker Average Episodes Reward / Evaluation')
    plt.ylabel('Cum Hack')

    plt.figure()
    plt.plot(df21["Step"], df21["Value"], color='red')
    plt.plot(df22["Step"], df22["Value"], color='blue')
    plt.plot(df23["Step"], df23["Value"], color='green')
    #plt.fill_between(df11["Step"], df1.min(axis=1), df1.max(axis=1), 1)
    #plt.title('Attacker Average Episodes Reward / Evaluation')
    plt.ylabel('Cum Hack')

    plt.figure()
    plt.plot(df31["Step"], df31["Value"], color='red')
    plt.plot(df32["Step"], df32["Value"], color='blue')
    plt.plot(df33["Step"], df33["Value"], color='green')
    plt.plot(df34["Step"], df34["Value"], color='orange')
    #plt.fill_between(df11["Step"], df1.min(axis=1), df1.max(axis=1), 1)
    #plt.title('Attacker Average Episodes Reward / Evaluation')
    plt.ylabel('Cum Hack')

    plt.figure()
    plt.plot(df41["Step"], df41["Value"], color='red')
    plt.plot(df42["Step"], df42["Value"], color='blue')
    plt.plot(df43["Step"], df43["Value"], color='green')
    plt.plot(df44["Step"], df44["Value"], color='orange')
    #plt.fill_between(df11["Step"], df1.min(axis=1), df1.max(axis=1), 1)
    #plt.title('Attacker Average Episodes Reward / Evaluation')
    plt.ylabel('Cum Hack')


    #plt.subplot(3, 2, 2)
    #plt.plot(df21["Step"], hdf1.mean(axis=1), color='red')
    #plt.fill_between(df11["Step"], hdf1.min(axis=1), hdf1.max(axis=1), 1)
    #plt.title('Attacker Cumulative Hack Probability / Evaluation')
    #plt.ylabel('Cum Hack')


    #plt.subplot(3, 1, 2)
    #plt.plot(df21["Step"], df2.mean(axis=1), color='red')
    #plt.plot(df21["Step"], df2.mean(axis=1), color='blue')
    #plt.fill_between(df21["Step"], df2.min(axis=1), df2.max(axis=1), 1)
    #plt.ylabel('Avg Reward')

    #plt.subplot(3, 2, 4)
    #plt.plot(df21["Step"], hdf2.mean(axis=1), color='red')
    #plt.fill_between(df21["Step"], hdf2.min(axis=1), hdf2.max(axis=1), 1)
    #plt.ylabel('Cum Hack')


    #plt.subplot(3, 1, 3)
    #plt.plot(df21["Step"], df3.mean(axis=1), color='red')
    #plt.fill_between(df31["Step"], df3.min(axis=1), df3.max(axis=1), 1)
    #plt.xlabel('Steps')
    #plt.ylabel('Avg Reward')

    ########################################################################3

    #plt.subplot(3, 2, 6)
    #plt.plot(df21["Step"], hdf3.mean(axis=1), color='red')
    #plt.fill_between(df31["Step"], hdf3.min(axis=1), hdf3.max(axis=1), 1)
    #plt.xlabel('Steps')
    #plt.ylabel('Cum Hack')

    plt.show()
