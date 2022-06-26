import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
fontP = FontProperties()

dbs=["agedb", "lfw", "calfw", "cplfw", "cfp","IJB-B", "IJB-C"]

for db in dbs:
    if (db=="agedb"):
        accuracies=[98.33, 97.95,  96.43,
                    98.08, 97.97,  97.43,
                    97.13,  97.07, 96.62,
                    95.62, 94.37, 91.77]

        params = [261.22, 65.31,
                  49.01,
                    174.68, 43.67, 32.77,
                    96.22, 24.1,  18.1,
                  4.21,   1.1,  0.79]
        nets = ["ResNet100",
                "ResNet100(w8a8)",
                "ResNet100(w6a6)",
                "ResNet50",
                "ResNet50(w8a8)",
                "ResNet50(w6a6)",
                "ResNet18",
                "ResNet18(w8a8)",
                "ResNet18(w6a6)",
                "MobileFaceNet",
                "MobileFaceNet(w8a8)",
                "MobileFaceNet(w6a6)"]

        marker = [ '+', '.', '*',
                   '+', '.', '*',
                   '+', '.', '*',
                   '+', '.', '*',
                   '+', '.', '*',
                   '+', '.', '*'
                   ]
        color = ['blue','blue','blue',
                 'green','green','green',
                 'red','red','red',
                 'darkgoldenrod','darkgoldenrod','darkgoldenrod']
        save_path = "./agedb.pdf"
        plt.figure()
        fig, ax = plt.subplots()
        plt.ylabel("Accuracy (%) ",fontsize=26)
        plt.xlabel("Size (MB)",fontsize=26)
        plt.ylim([90, 98.50])
        plt.xlim([-2, 267])
    elif(db=="lfw"):
        accuracies = [99.83,  99.8, 99.45,
                    99.80, 99.78, 99.68,
                      99.67,
                     99.55,  99.55,
                      99.47,
                    99.35,  99.08]

        save_path = "./lfw.pdf"
        plt.figure()
        fig, ax = plt.subplots()
        plt.ylabel("Accuracy (%) ",fontsize=26)
        plt.xlabel("Size (MB)",fontsize=26)
        plt.ylim([98, 99.89])
        plt.xlim([-2, 267])
    elif(db=="calfw"):
        accuracies = [96.13,  96.02, 95.58,
                    96.1, 95.87,  95.7,
                      95.70,95.58, 95.32,
                    95.15, 94.78,  93.48]


        save_path = "./calfw.pdf"
        plt.figure()
        fig, ax = plt.subplots()
        plt.plot(accuracies, params, 'o')
        plt.ylabel("Accuracy (%) ",fontsize=26)
        plt.xlabel("Size (MB)",fontsize=26)
        plt.ylim([90, 97])
        plt.xlim([-2, 267])
    elif(db=="cplfw"):
        accuracies = [93.22,  92.9,  86.6,
                    92.43,  92.08,  90.38,
                    89.73,  89.53,  89.05,
                    87.98,  87.73,  84.85]


        save_path = "./cplfw.pdf"
        plt.figure()
        fig, ax = plt.subplots()
        plt.ylabel("Accuracy (%) ",fontsize=26)
        plt.xlabel("Size (MB)",fontsize=26)
        plt.ylim([82, 94.2])
        plt.xlim([-2, 267])
    elif(db=="cfp"):
        accuracies = [98.4,  98.14,  91.0,
                    98.01,  97.43,  95.17,
                    94.47,  94.04,  93.34,
                    91.59,  90.84,  87.64]
        save_path = "./cfp.pdf"
        plt.figure()
        fig, ax = plt.subplots()
        plt.ylabel("Accuracy (%) ",fontsize=26)
        plt.xlabel("Size (MB)",fontsize=26)
        plt.ylim([84, 99])
        plt.xlim([-2, 267])

    elif(db=="IJB-B"):
        accuracies = [95.25, 94.74,  85.06,
                    94.19,  93.67,  89.44,
                    91.64,  91.01,  90.38,
                    88.54,  86.98,  80.58]
        save_path = "./ijbb.pdf"
        plt.figure()
        fig, ax = plt.subplots()
        plt.ylabel("TAR at FAR1e–4 ",fontsize=26)
        plt.xlabel("Size (MB)",fontsize=26)
        plt.ylim([78, 97])
        plt.xlim([-2, 267])
    elif(db=="IJB-C"):
        accuracies = [96.5,  96.09,  87.0,
                    95.74,  95.18,  90.72,
                      93.56,  92.87,  92.36,
                      90.88,
                    89.21,  82.94]


        save_path = "./ijbc.pdf"
        plt.figure()
        fig, ax = plt.subplots()
        plt.ylabel("TAR at FAR1e–4 ",fontsize=26)
        plt.xlabel("Size (MB)",fontsize=26)
        plt.ylim([80, 97])
        plt.xlim([-2, 267])

    p=[]
    for i in range(len(accuracies)):
        #if "ours" in nets[i]:
        # plt.plot(params[i], accuracies[i], marker[i],markersize=16,markeredgecolor='red',label=nets[i])
        #else:
            plt.plot(params[i], accuracies[i], marker[i],color=color[i], markersize=16,label=nets[i])

    plt.grid()
    plt.tight_layout()

    plt.legend(numpoints=1, loc='lower right',fontsize=12,ncol=2)
    plt.savefig(save_path, format='pdf', dpi=600)
    plt.close()
