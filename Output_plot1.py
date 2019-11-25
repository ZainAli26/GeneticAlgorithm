import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
# https://realpython.com/python-matplotlib-guide/


## Display the logs
def Analysis_plot(TrainTiab_PHIF, TestTiab_PHIF, ValidationTiab_PHIF, TrainGA_PHIF, TestGA_PHIF, ValidationGA_PHIF, TEPD, Depth, DEPT):
    fig = plt.figure(figsize =(10 ,10))
    fig.suptitle("Fracture Porosity Analysis", fontsize=20)
    fig.subplots_adjust(top=0.85, bottom=0.03, wspace=0.5)

    ### General setting For Training data
    ax1 = fig.add_subplot(131)
    ax1.text(0.5, 1.08, 'Training data', horizontalalignment='center', fontsize=10, transform = ax1.transAxes)
    ax1.set_ylim(TEPD[0], Depth[0])
    ax1.invert_yaxis() # invert y axis

    major_y_spacing = (Depth[0]- TEPD[0])/10
    majoryLocator = MultipleLocator(major_y_spacing)
    ax1.yaxis.set_major_locator(majoryLocator)
    ax1.grid(True, which='major', axis='y', color='black', linestyle='-', linewidth=1)

    minor_y_spacing = (Depth[0]- TEPD[0])/50
    minoryLocator = MultipleLocator(minor_y_spacing)
    ax1.yaxis.set_minor_locator(minoryLocator)
    ax1.tick_params(axis='y', labelsize=8)
    ax1.grid(True, which='minor', axis='y' ,alpha = 0.5, color='grey', linestyle='--')
    ax1.set_ylabel('Depth (m)' ,color='black', fontsize = 10)

    ### 1st tack: For Training data

    # PHIF and GA_PHIF track

    ax1.set_xlabel("PHIF[Fraction]", color='black', fontsize = 8)
    ax1.xaxis.set_label_position('top')
    ax1.plot(TrainTiab_PHIF, TEPD, label='PHIF', color= 'black', linewidth=1, linestyle = '-')
    ax1.plot(TrainGA_PHIF, TEPD, label='GA_PHIF', color= 'red', linewidth=1, linestyle = '-')

    ax1.set_xlim(0.45, -0.15)
    ax1.invert_xaxis()
    min_x = -0.15
    max_x = 0.45

    ax1.xaxis.tick_top()
    ax1.xaxis.set_ticks(np.arange(min_x, max_x, 0.299999))
    ax1.grid(True, which='major', axis='x', color = 'black', linestyle='-', linewidth=1.5)

    minor_spacing_PHIF = 0.03
    minorLocatorPHIF = MultipleLocator(minor_spacing_PHIF)
    ax1.xaxis.set_minor_locator(minorLocatorPHIF)

    ax1.spines['top'].set_position(('outward', 0))
    ax1.tick_params(axis='x', colors='black', labelsize=8)
    ax1.grid(True, which='minor', axis='x', alpha=0.2, color='grey', linestyle='--')

    ax1.legend(loc='upper right', fontsize = 10)

    ###########################################
    ###########################################
    ###########################################

    ### General setting For Validation data
    ax3 = fig.add_subplot(132)
    ax3.text(0.5, 1.08, 'Validation data', horizontalalignment='center', fontsize=10, transform=ax3.transAxes)
    ax3.set_ylim(Depth[0], Depth[-1])
    ax3.invert_yaxis()  # invert y axis

    major_y_spacing = (Depth[-1] - Depth[0]) / 10
    majoryLocator = MultipleLocator(major_y_spacing)
    ax3.yaxis.set_major_locator(majoryLocator)
    ax3.grid(True, which='major', axis='y', color='black', linestyle='-', linewidth=1)

    minor_y_spacing = (Depth[-1] - Depth[0]) / 50
    minoryLocator = MultipleLocator(minor_y_spacing)
    ax3.yaxis.set_minor_locator(minoryLocator)
    ax3.tick_params(axis='y', labelsize=8)
    ax3.grid(True, which='minor', axis='y', alpha=0.5, color='grey', linestyle='--')
    # ax1.set_ylabel('Depth (m)', color='black', fontsize=10)


    ### 1st tack: For Validation data

    # PHIF and GA_PHIF track

    ax3.set_xlabel("PHIF[Fraction]", color='black', fontsize=8)
    ax3.xaxis.set_label_position('top')
    ax3.plot(ValidationTiab_PHIF, Depth, label='PHIF', color='black', linewidth=1, linestyle='-')
    ax3.plot(ValidationGA_PHIF, Depth, label='GA_PHIF', color='red', linewidth=1, linestyle='-')

    ax3.set_xlim(0.45, -0.15)
    ax3.invert_xaxis()
    min_x = -0.15
    max_x = 0.45

    ax3.xaxis.tick_top()
    ax3.xaxis.set_ticks(np.arange(min_x, max_x, 0.299999))
    ax3.grid(True, which='major', axis='x', color='black', linestyle='-', linewidth=1.5)

    minor_spacing_PHIF = 0.03
    minorLocatorPHIF = MultipleLocator(minor_spacing_PHIF)
    ax3.xaxis.set_minor_locator(minorLocatorPHIF)

    ax3.spines['top'].set_position(('outward', 0))
    ax3.tick_params(axis='x', colors='black', labelsize=8)
    ax3.grid(True, which='minor', axis='x', alpha=0.2, color='grey', linestyle='--')

    ax3.legend(loc='upper right', fontsize=10)

    ###########################################
    ###########################################
    ###########################################

    ### General setting For Testing data
    ax2 = fig.add_subplot(133)
    ax2.text(0.5, 1.08, 'Testing data', horizontalalignment='center', fontsize=10, transform = ax2.transAxes)
    ax2.set_ylim(DEPT[0], DEPT[-1])
    ax2.invert_yaxis() # invert y axis

    major_y_spacing = (DEPT[-1]-DEPT[0])/10
    majoryLocator = MultipleLocator(major_y_spacing)
    ax2.yaxis.set_major_locator(majoryLocator)
    ax2.grid(True, which='major', axis='y', color='black', linestyle='-', linewidth=1)

    minor_y_spacing = (DEPT[-1]-DEPT[0])/50
    minoryLocator = MultipleLocator(minor_y_spacing)
    ax2.yaxis.set_minor_locator(minoryLocator)
    ax2.tick_params(axis='y', labelsize=8)
    ax2.grid(True, which='minor', axis='y' ,alpha = 0.5, color='grey', linestyle='--')
    # ax2.set_ylabel('Depth (m)',color='black', fontsize = 10)

    ### 1st tack: For Testing data

    # PHIF and GA PHIF track

    ax2.set_xlabel("PHIF[Fraction]", color='black', fontsize = 8)
    ax2.xaxis.set_label_position('top')
    ax2.plot(TestTiab_PHIF, DEPT, label='PHIF', color= 'black', linewidth=1, linestyle = '-')
    ax2.plot(TestGA_PHIF, DEPT, label='GA_PHIF', color= 'green', linewidth=1, linestyle = '-')

    ax2.set_xlim(0.45, -0.15)
    ax2.invert_xaxis()
    min_x = -0.15
    max_x = 0.45

    ax2.xaxis.tick_top()
    ax2.xaxis.set_ticks(np.arange(min_x, max_x, 0.299999))
    ax2.grid(True, which='major', axis='x', color = 'black', linestyle='-', linewidth=1.5)

    minor_spacing_PHIF = 0.03
    minorLocatorPHIF = MultipleLocator(minor_spacing_PHIF)
    ax2.xaxis.set_minor_locator(minorLocatorPHIF)

    ax2.spines['top'].set_position(('outward', 0))
    ax2.tick_params(axis='x', colors='black', labelsize=8)
    ax2.grid(True, which='minor', axis='x', alpha=0.2, color='grey', linestyle='--')

    ax2.legend(loc='upper right', fontsize = 10)

## Display the logs
from matplotlib import gridspec


def cost_plot(generation, bestcost_train, bestcost_validation,bestcost_test):
    fig = plt.figure(figsize=(7,7))
    fig.suptitle('Genetic Algorithm',fontsize=14)
    # plt.show()

    fig.subplots_adjust(top=0.85, bottom=0.25)
    gs = gridspec.GridSpec(1, 1)  # (rows and columns)
    ax1 = fig.add_subplot(gs[0,0])
    ax1.text(0.5, 1.08, 'Cost vs. Generations', horizontalalignment='center', fontsize=10, transform = ax1.transAxes)
    ax1.set_yscale('log')
    ax1.set_ylim(ymin=0.1)
    # ax1.set_ylim(round(np.min(cost),1), round(np.max(cost),1))
    ax1.autoscale(enable=True, axis='y')

    ax1.set_ylabel('Cost (Fraction)', color='black', fontsize=15)
    # major_y_spacing = round((round(np.max(cost),1)-0),0)/5
    # majoryLocator = MultipleLocator(major_y_spacing)
    # ax1.yaxis.set_major_locator(majoryLocator)
    ax1.grid(True, which='major', axis='y', color='black', linestyle='-', linewidth=1)

    # minor_y_spacing = round((round(np.max(cost),1)-0),0)/25
    # minoryLocator = MultipleLocator(minor_y_spacing)
    # ax1.yaxis.set_minor_locator(minoryLocator)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.grid(True, which='minor', axis='y' ,alpha = 0.5, color='grey', linestyle='--')

    # Cost X limits

    ax1.set_xlabel("Generations", color='black', fontsize = 15)
    ax1.xaxis.set_label_position('bottom')

    ax1.plot(generation, bestcost_train, label='Training data', color= 'red', linewidth=2, linestyle = '-')
    ax1.plot(generation, bestcost_validation, label='Validation data', color='limegreen', linewidth=2, linestyle='--')
    ax1.plot(generation, bestcost_test, label='Test data', color='blue', linewidth=2, linestyle='--')

    ax1.set_xlim(0, generation[-1])
    min_x = 0
    max_x = generation[-1]

    # ax1.xaxis.tick_top()
    ax1.xaxis.set_ticks(np.arange(min_x, max_x, 50))
    # ax1.autoscale(enable=True, axis='both')
    ax1.grid(True, which='major', axis='x', color = 'black', linestyle='-', linewidth=0.5)

    ax1.spines['top'].set_position(('outward', 0))
    ax1.tick_params(axis='x', colors='black', labelsize=15)
    ax1.grid(True, which='minor', axis='x', alpha=0.2, color='grey', linestyle='--')

    ax1.legend(loc='upper right', fontsize = 10)