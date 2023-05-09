import numpy as np
import matplotlib.pylab as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker
from lib.functions import *


########## CHAOS ##########
class CHAOS():

    def __init__(self, rcParams_dict):
        for key in rcParams_dict.keys():
            plt.rcParams[str(key)] = rcParams_dict[str(key)]      

    def figure02(self,
                data, t_data,
                data_at1, n_shift_at1, n_plot_at1,
                data_at2, n_shift_at2, n_plot_at2,
                parameter_data=None, 
                figsize=(25, 5), width_ratios=[2, 1, 1], wspace=0.4,
                xlabel=r'$\mu$'+' [l/min]', ylabel='Frequency [Hz]', nperseg=2000, cmaplim=(-220, 80), cmap='jet',  xlim=(0.45, 1.0), ylim=(0, 500), 
                title_at1='parameter1', title_at2='parameter2', linestyle_at='-', c_at='k', lw_at=2, xlabel_at=r'$x(t)$', ylabel_at=r'$x(t-\tau)$', same_lim=True,
                panel_list = ['(a)', '(b)', '(c)'], panel_xy_list=[(0.35, 1.), (0.2, 1.), (0.2, 1)], panel_fontsize=40, 
                save_filename=None):
        
        fig = plt.figure(figsize=figsize)
        spec = gridspec.GridSpec(ncols=3, nrows=1,
                         width_ratios=width_ratios,
                         wspace=wspace
                         )
        
        ax0 = fig.add_subplot(spec[0])
        if parameter_data.all()==None:
            parameter_data = t_data
        dt = t_data[1] - t_data[0]
        freq, t_stft, intens = short_time_fourier_transform(data, dt, nperseg)
        p = np.linspace(parameter_data[0], parameter_data[-1], t_stft.shape[0])
        db = 10*np.log(np.abs(intens))
        db_plt = np.where(db<cmaplim[0], cmaplim[0], db)
        db_plt = np.where(db_plt>cmaplim[1], cmaplim[1], db_plt)
        ax0.pcolormesh(p, freq, db_plt, cmap=cmap)
        ax0.set_xlim(xlim)
        ax0.set_ylim(ylim)
        ax0.set_xlabel(xlabel)
        ax0.set_ylabel(ylabel)
        fig.text(ax0.get_position().x1-panel_xy_list[0][0], ax0.get_position().y1-panel_xy_list[0][1], s=panel_list[0], fontsize=panel_fontsize)

        ax2 = fig.add_subplot(spec[2])
        ax2.set_title(title_at2, loc='center')
        ax2.plot(data_at2[n_shift_at2:][:n_plot_at2], data_at2[:-n_shift_at2][:n_plot_at2], linestyle=linestyle_at, c=c_at, lw=lw_at)
        ax2.set_xlabel(xlabel_at)
        ax2.set_ylabel(ylabel_at)
        ax2.set_aspect('equal', 'datalim')
        fig.text(ax2.get_position().x1-panel_xy_list[2][0], ax2.get_position().y1-panel_xy_list[2][1], s=panel_list[2], fontsize=panel_fontsize)

        ax1 = fig.add_subplot(spec[1])
        ax1.set_title(title_at1, loc='center')
        ax1.plot(data_at1[n_shift_at1:][:n_plot_at1], data_at1[:-n_shift_at1][:n_plot_at1], linestyle=linestyle_at, c=c_at, lw=lw_at)
        ax1.set_xlabel(xlabel_at)
        ax1.set_ylabel(ylabel_at)
        ax1.set_aspect('equal', 'datalim')
        if same_lim:
            ax1.set_xlim(ax2.get_xlim())
            ax1.set_ylim(ax2.get_ylim())
        fig.text(ax1.get_position().x1-panel_xy_list[1][0], ax1.get_position().y1-panel_xy_list[1][1], s=panel_list[1], fontsize=panel_fontsize)

        plt.tight_layout()
        if save_filename==None:
            plt.show()
        else:
            plt.savefig(save_filename)

    def figure02_2(self,
                data, t_data,
                data_at1, n_shift_at1, n_plot_at1,
                data_at2, n_shift_at2, n_plot_at2,
                data_bf, parameter_data_bf, parameter_lim_bf,
                parameter_data=None, 
                figsize=(25, 5), width_ratios=[2, 1, 2, 1], wspace=0.4, hspace=0.1,
                xlabel=r'$\mu$'+' [l/min]', ylabel='Frequency [Hz]', nperseg=2000, cmaplim=(-220, 80), cmap='jet',  xlim=(0.45, 1.0), ylim=(0, 500), 
                title_at1='parameter1', title_at2='parameter2', title_size=10, linestyle_at='-', c_at='k', lw_at=2, xlabel_at=r'$x(t)$', ylabel_at=r'$x(t-\tau)$', same_lim=True,
                marker_bf='.', c_bf='b', s_bf=50, xlabel_bf=r'$\mu$'+' [l/min]', ylabel_bf=r'$y_l$',
                panel_list = ['(a)', '(b)', '(c)', '(d)'], panel_xy_list=[(0.35, 1.), (0.2, 1.), (0.35, 1.), (0.2, 1)], panel_fontsize=40, 
                save_filename=None):
        
        fig = plt.figure(figsize=figsize)
        spec = gridspec.GridSpec(ncols=2, nrows=2,
                         width_ratios=width_ratios,
                         wspace=wspace,
                         hspace=hspace
                         )
        
        ax0 = fig.add_subplot(spec[0])
        if parameter_data.all()==None:
            parameter_data = t_data
        dt = t_data[1] - t_data[0]
        freq, t_stft, intens = short_time_fourier_transform(data, dt, nperseg)
        p = np.linspace(parameter_data[0], parameter_data[-1], t_stft.shape[0])
        db = 10*np.log(np.abs(intens))
        db_plt = np.where(db<cmaplim[0], cmaplim[0], db)
        db_plt = np.where(db_plt>cmaplim[1], cmaplim[1], db_plt)
        ax0.pcolormesh(p, freq, db_plt, cmap=cmap)
        ax0.set_xlim(xlim)
        ax0.set_ylim(ylim)
        ax0.set_xlabel(xlabel)
        ax0.set_ylabel(ylabel)
        fig.text(ax0.get_position().x1-panel_xy_list[0][0], ax0.get_position().y1-panel_xy_list[0][1], s=panel_list[0], fontsize=panel_fontsize)

        ax2 = fig.add_subplot(spec[3])
        ax2.set_title(title_at2, loc='center',fontsize=title_size)
        ax2.plot(data_at2[n_shift_at2:][:n_plot_at2], data_at2[:-n_shift_at2][:n_plot_at2], linestyle=linestyle_at, c=c_at, lw=lw_at)
        ax2.set_xlabel(xlabel_at)
        ax2.set_ylabel(ylabel_at)
        ax2.set_aspect('equal', 'datalim')
        fig.text(ax2.get_position().x1-panel_xy_list[3][0], ax2.get_position().y1-panel_xy_list[3][1], s=panel_list[3], fontsize=panel_fontsize)

        ax3 = fig.add_subplot(spec[2])
        for i in range(len(data_bf)):  
            ax3.scatter(np.full(data_bf[i].shape, parameter_data_bf[i]), data_bf[i], marker=marker_bf, c=c_bf, s=s_bf)
        ax3.set_xlim(parameter_lim_bf)
        ax3.set_xlabel(xlabel_bf)
        ax3.set_ylabel(ylabel_bf)
        fig.text(ax3.get_position().x1-panel_xy_list[2][0], ax3.get_position().y1-panel_xy_list[2][1], s=panel_list[2], fontsize=panel_fontsize)

        ax1 = fig.add_subplot(spec[1])
        ax1.set_title(title_at1, loc='center',fontsize=title_size)
        ax1.plot(data_at1[n_shift_at1:][:n_plot_at1], data_at1[:-n_shift_at1][:n_plot_at1], linestyle=linestyle_at, c=c_at, lw=lw_at)
        ax1.set_xlabel(xlabel_at)
        ax1.set_ylabel(ylabel_at)
        ax1.set_aspect('equal', 'datalim')
        if same_lim:
            ax1.set_xlim(ax2.get_xlim())
            ax1.set_ylim(ax2.get_ylim())
        fig.text(ax1.get_position().x1-panel_xy_list[1][0], ax1.get_position().y1-panel_xy_list[1][1], s=panel_list[1], fontsize=panel_fontsize)

        plt.tight_layout()
        if save_filename==None:
            plt.show()
        else:
            plt.savefig(save_filename, bbox_inches="tight")
        

    def figure03(self, 
                data, model, t, lyapunov_exponents, 
                figsize=(35, 5), width_ratios=[4, 4, 7, 8], wspace=0.4,
                n_shift=25, n_initdel=2000, n_plt=3000, same_lim=True,
                lw_data=3, lw_model=4,
                freq_lim=(50, 350),
                n_dim=4, lyapunov_lim=(-110, 20), 
                panel_list = ['(a)', '(b)', '(c)', '(d)'], panel_xy_list=[(0.14, 1.), (0.14, 1.), (0.21, 1), (0.25, 1)], panel_fontsize=40, 
                save_filename=None):
        spec = gridspec.GridSpec(ncols=4, nrows=1, width_ratios=width_ratios, wspace=wspace)
        fig = plt.figure(figsize=figsize)
        freq_data, amp_data = fft(data, t)
        freq_model, amp_model = fft(model, t)

        ax0 = fig.add_subplot(spec[0])
        ax0.set_title('Exp', loc='center')
        ax0.plot(data[n_initdel:][n_shift:][:n_plt], data[n_initdel:][:-n_shift][:n_plt], linestyle='-', c='k', lw=lw_data)
        ax0.set_xlabel(r'$x(t)$')
        ax0.set_ylabel(r'$x(t-\tau)$')
        ax0.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax0.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax0.set_aspect('equal', 'datalim')
        fig.text(ax0.get_position().x1-panel_xy_list[0][0], ax0.get_position().y1-panel_xy_list[0][1], s=panel_list[0], fontsize=panel_fontsize)

        ax1 = fig.add_subplot(spec[1])
        ax1.set_title('Model', loc='center')
        ax1.plot(model[n_initdel:][n_shift:][:n_plt], model[n_initdel:][:-n_shift][:n_plt], linestyle='-', c='r', lw=lw_model)
        if same_lim:
            ax1.set_xlim(ax0.get_xlim())
            ax1.set_ylim(ax0.get_ylim())
        ax1.set_xlabel(r'$x(t)$')
        ax1.set_ylabel(r'$x(t-\tau)$')
        ax1.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax1.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax1.set_aspect('equal', 'datalim')
        fig.text(ax1.get_position().x1-panel_xy_list[1][0], ax1.get_position().y1-panel_xy_list[1][1], s=panel_list[1], fontsize=panel_fontsize)

        ax2 = fig.add_subplot(spec[2])
        ax2.plot(freq_data, amp_data, lw=5, c='k', label='Exp')
        ax2.plot(freq_model, amp_model, '--', lw=4, c='r', label='Model')
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Power Spectral Density')
        ax2.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%d'))
        ax2.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax2.set_xlim(freq_lim)
        ax2.legend(frameon=False)
        fig.text(ax2.get_position().x1-panel_xy_list[2][0], ax2.get_position().y1-panel_xy_list[2][1], s=panel_list[2], fontsize=panel_fontsize)

        ax3 = fig.add_subplot(spec[3])
        ax3.axhline(y=0, xmin=0, xmax=n_dim+1, linestyle='dashed', c='b', lw=4)
        ax3.plot(np.arange(1, n_dim+1), lyapunov_exponents[:n_dim], linestyle='-', c='r', 
                lw=4, marker='o', markersize=15)
        ax3.grid()
        ax3.set_ylim(lyapunov_lim)
        ax3.set_xlabel('Dimension')
        ax3.set_ylabel('Lyapunov Exponents')
        fig.text(ax3.get_position().x1-panel_xy_list[3][0], ax3.get_position().y1-panel_xy_list[3][1], s=panel_list[3], fontsize=panel_fontsize)

        plt.tight_layout()
        if save_filename==None:
            plt.show()
        else:
            plt.savefig(save_filename, bbox_inches="tight")

    def figure04_snapshots(self,
                        data, t_data, position, width, c_line='r', lw_line=5, start=0, step=10, start_t_is_0=True,
                        title='Vocal Fold',
                        figsize=(17, 2), n_shots=10, wspace=0.6, aspect=1.3,
                        gamma=1,
                        panel = '(a)', panel_xy=(0.14, 1.), panel_fontsize=40, 
                        save_filename=None):
        if gamma!=1:
            data = gamma_correction(data, gamma)
        
        fig = plt.figure(figsize=figsize)
        spec = gridspec.GridSpec(ncols=n_shots, nrows=1,
                         #height_ratios =[1, 1],
                         wspace=wspace
                         )
        x = data.shape[1]
        y = data.shape[2]
        a = position[0]
        b = position[1]
        for i in range(n_shots):
            ax = fig.add_subplot(spec[i])
            if i==0:
                ax.set_title(title, loc='left')
            ax.imshow(np.rot90(data[start+i*step], -1), cmap='Greys_r', norm=Normalize(vmin=0, vmax=255), aspect=aspect)
            if i==0:
                ax.plot([x-b, x-b], [a-int(width/2), a+int(width/2)], color=c_line, linewidth=lw_line)
            ax.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
            if i==0:
                fig.text(ax.get_position().x1-panel_xy[0], ax.get_position().y1-panel_xy[1], s=panel, fontsize=panel_fontsize)
            if start_t_is_0:
                ax.set_xlabel('{:.1f}'.format(i*step*(t_data[1]-t_data[0])*1000)+' [ms]')
            else:
                ax.set_xlabel('{:.1f}'.format(t_data[start+i*step]*1000)+' [ms]')
        plt.tight_layout()
        if save_filename==None:
            plt.show()
        else:
            plt.savefig(save_filename, bbox_inches="tight")

    def figure04_linescannning(self, 
                            data, t_data, position, width, start=0, n_frame=5000, start_t_is_0=True,
                            title='Vocal Fold',
                            figsize=(17, 2), aspect=1.3,
                            gamma=1,
                            panel = '(a)', panel_xy=(0.14, 1.), panel_fontsize=40, 
                            save_filename=None):
        if gamma!=1:
            data = gamma_correction(data, gamma)
        data_linescanned = line_scanning(data, position, width)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_title(title, loc='left')
        ax.imshow(data_linescanned[start:][:n_frame].T, cmap='Greys_r', norm=Normalize(vmin=0, vmax=255), aspect=aspect)
        ax.tick_params(labelleft=False, left=False, labelbottom=True, bottom=True)
        xlocs, xlabs = plt.xticks()
        frame_num = np.arange(0, n_frame, 1)
        idx = np.where(np.isin(frame_num, xlocs))
        if start_t_is_0:
            xlocs_new = np.round(np.arange(0, n_frame, 1)*(t_data[1]-t_data[0])*1000, decimals=3)
            plt.xticks(xlocs[1:-1], xlocs_new[xlocs[1:-1].astype(np.int64)])
        else:
            xlocs_new = np.round(t_data[start:start+n_frame]*1000, decimals=3)
            plt.xticks(xlocs[1:-1], xlocs_new[xlocs[1:-1].astype(np.int64)])
        ax.set_xlabel('Time [ms]')
        fig.text(ax.get_position().x1-panel_xy[0], ax.get_position().y1-panel_xy[1], s=panel, fontsize=panel_fontsize)
        plt.tight_layout()
        if save_filename==None:
            plt.show()
        else:
            plt.savefig(save_filename, bbox_inches="tight")

    def figure05(self,
                data1_list, t_data1_list, 
                data2_list, t_data2_list,
                n_sample=10000,
                label1='data1', label2='data2',
                linestyle1='-', linestyle2='--', 
                c1='b', c2='r', 
                lw1=3, lw2=2, 
                figsize=(40, 5), 
                wspace=0.4,
                freq_lim=(0, 500),
                freq_ylim_list=[(None, None), (None, None), (None, None)],
                title_list=[None, None],
                legend_fontsize=20,
                panel_list = ['(a)', '(b)'], panel_xy_list=[(0.14, 1.), (0.14, 1.)], panel_fontsize=40, 
                save_filename=None,):
        fig = plt.figure(figsize=figsize)
        spec = gridspec.GridSpec(ncols=3, nrows=1,
                                wspace=wspace
                                )
        for i in range(3):
            data1 = data1_list[i][:n_sample]
            t_data1 = t_data1_list[i][:n_sample]
            data2 =data2_list[i][:n_sample]
            t_data2 = t_data2_list[i][:n_sample]
            freq_data1, amp_data1 = fft(data1, t_data1)
            freq_data2, amp_data2 = fft(data2, t_data2)
            ax = fig.add_subplot(spec[i])
            ax.set_title(title_list[i], loc='center')
            ax.plot(freq_data1, amp_data1, linestyle=linestyle1, lw=lw1, c=c1, label=label1)
            ax.plot(freq_data2, amp_data2, linestyle=linestyle2, lw=lw2, c=c2, label=label2)
            ax.legend(loc='upper right', frameon=False, fontsize=legend_fontsize)
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Power Spectral Density')
            ax.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%d'))
            ax.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
            ax.set_xlim(freq_lim)
            ax.set_ylim(freq_ylim_list[i])
            fig.text(ax.get_position().x1-panel_xy_list[i][0], ax.get_position().y1-panel_xy_list[i][1], s=panel_list[i], fontsize=panel_fontsize)
        plt.tight_layout()
        if save_filename==None:
            plt.show()
        else:
            plt.savefig(save_filename, bbox_inches="tight")

    def figure06(self,
                 data1_list, data1_parameter_list, 
                 data2,
                 conversion_param=40., 
                 figsize=(25, 5), width_ratios=[1, 1, 0.5], wspace=0.3,
                 title1=None, title2=None, title3=None,
                 marker1='.', marker2='.', marker3='.',
                 c1='b', c2='r', c3='r', 
                 s1=50, s2=50, s3=50,
                 xlim1=(None, None), xlim2=(None, None), xlim3=(None, None),
                 ylim1=(None, None), ylim2=(None, None), ylim3=(None, None),
                 xlabel1=r'$mu $'+'[l/min]', xlabel2=r'$mu $'+'[l/min]', xlabel3=r'$mu $'+'[l/min]', 
                 ylabel1=r'$y_l$', ylabel2=r'$y_l$', ylabel3=r'$y_l$', 
                 panel_list = ['(a)', '(b)', '(c)'], panel_xy_list=[(0.14, 1.), (0.14, 1.), (0.14, 1.)], panel_fontsize=40, 
                save_filename=None,
                 ):
        spec = gridspec.GridSpec(ncols=3, nrows=1,
                                 width_ratios=width_ratios,
                                 wspace=wspace)
        fig = plt.figure(figsize=figsize)

        ax0 = fig.add_subplot(spec[0])
        ax0.set_title(title1, loc='center')
        for i in range(len(data1_list)):  
            ax0.scatter(np.full(data1_list[i].shape, data1_parameter_list[i]*conversion_param), data1_list[i], marker=marker1, c=c1, s=s1)
        ax0.tick_params(labelleft=False, left=False, labelbottom=True, bottom=True)
        ax0.set_xlim(xlim1)
        ax0.set_ylim(ylim1)
        ax0.set_xlabel(xlabel1)
        ax0.set_ylabel(ylabel1)
        fig.text(ax0.get_position().x1-panel_xy_list[0][0], ax0.get_position().y1-panel_xy_list[0][1], s=panel_list[0], fontsize=panel_fontsize)

        ax1 = fig.add_subplot(spec[1])
        ax1.set_title(title2, loc='center')
        ax1.scatter(data2[:, 1]*conversion_param, data2[:, 0], marker=marker2, c=c2, s=s2)
        ax1.tick_params(labelleft=False, left=False, labelbottom=True, bottom=True)
        ax1.set_xlim(xlim2)
        ax1.set_ylim(ylim2)
        ax1.set_xlabel(xlabel2)
        ax1.set_ylabel(ylabel2)
        fig.text(ax1.get_position().x1-panel_xy_list[1][0], ax1.get_position().y1-panel_xy_list[1][1], s=panel_list[1], fontsize=panel_fontsize)

        ax2 = fig.add_subplot(spec[2])
        ax2.set_title(title3, loc='center')
        ax2.scatter(data2[:, 1]*conversion_param, data2[:, 0], marker=marker3, c=c3, s=s3)
        ax2.tick_params(labelleft=False, left=False, labelbottom=True, bottom=True)
        ax2.set_xlim(xlim3)
        ax2.set_ylim(ylim3)
        ax2.set_xlabel(xlabel3)
        ax2.set_ylabel(ylabel3)
        fig.text(ax2.get_position().x1-panel_xy_list[2][0], ax2.get_position().y1-panel_xy_list[2][1], s=panel_list[2], fontsize=panel_fontsize)

        plt.tight_layout()
        if save_filename==None:
            plt.show()
        else:
            plt.savefig(save_filename, bbox_inches="tight")


########## NLP ##########
class NLP():

    def __init__(self, rcParams_dict):
        for key in rcParams_dict.keys():
            plt.rcParams[str(key)] = rcParams_dict[str(key)]      

    def figure01(self, 
                data, t_data, ms=True, n_sample0=3000, n_sample1=7000, n_sample2=1500, freq_lim=(50, 300), n_shift=20, 
                figsize=(30, 5), width_ratios=[3, 2, 1.4], wspace=0.4,
                title0=None, title1=None, title2=None, title_loc0='left', title_loc1='left', title_loc2='left', 
                xlabel0='Time [ms]', ylabel0=r'$x(t)$', linestyle0='-', c0='k', lw0='3',
                xlabel1='Frequency [Hz]', ylabel1='Power Spectral Density', linestyle1='-', c1='k', lw1='3',
                xlabel2=r'$x(t)$', ylabel2=r'$x(t-\tau)$', linestyle2='-', c2='k', lw2='3',
                panel_list = ['(a)', '(b)', '(c)'], panel_xy_list=[(0.33, 0.93), (0.23, 0.93), (0.17, 0.93)], panel_fontsize=40, 
                save_png=None, save_eps=None):
        fig = plt.figure(figsize=figsize)
        spec = gridspec.GridSpec(ncols=3, nrows=1,
                            width_ratios=width_ratios,
                            wspace=wspace
                            )
        if ms:
            plt_t = t_data*1000
        else:
            plt_t = t_data
        ax0 = fig.add_subplot(spec[0])
        ax0.set_title(title0, loc=title_loc0)
        ax0.plot(plt_t[:n_sample0], data[:n_sample0], linestyle=linestyle0, c=c0, lw=lw0)
        ax0.set_xlabel(xlabel0)
        ax0.set_ylabel(ylabel0)
        ax0.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%d'))
        ax0.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        fig.text(ax0.get_position().x1-panel_xy_list[0][0], ax0.get_position().y1-panel_xy_list[0][1], s=panel_list[0], fontsize=panel_fontsize)

        ax1 = fig.add_subplot(spec[1])
        freq_data, amp_data = fft(data[:n_sample1], t_data[:n_sample1])
        ax1.set_title(title1, loc=title_loc1)
        ax1.plot(freq_data, amp_data, linestyle=linestyle1, c=c1, lw=lw1)
        ax1.set_xlabel(xlabel1)
        ax1.set_ylabel(ylabel1)
        ax1.set_xlim(freq_lim)
        ax1.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%d'))
        ax1.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        fig.text(ax1.get_position().x1-panel_xy_list[1][0], ax1.get_position().y1-panel_xy_list[1][1], s=panel_list[1], fontsize=panel_fontsize)

        ax2 = fig.add_subplot(spec[2])
        ax2.set_title(title2, loc=title_loc2)
        ax2.plot(data[n_shift:][:n_sample2], data[:-n_shift][:n_sample2], linestyle=linestyle2, c=c2, lw=lw2)
        ax2.set_xlabel(xlabel2)
        ax2.set_ylabel(ylabel2)
        ax2.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax2.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax2.set_aspect('equal', 'datalim')
        fig.text(ax2.get_position().x1-panel_xy_list[2][0], ax2.get_position().y1-panel_xy_list[2][1], s=panel_list[2], fontsize=panel_fontsize)

        plt.tight_layout()
        if save_png==None:
            plt.show()
        else:
            plt.savefig(save_png+'.png', bbox_inches="tight")
        if save_png==None:
            return
        else:
            plt.savefig(save_eps+'.eps', bbox_inches="tight")

    def figure03(self,
                data, t_data, scanline=True, position=[50, 40], width=40, c_line='r', lw_line=4, start=80, step=30, start_t_is_0=True,
                title='Vocal fold',
                figsize=(20, 2), n_shots=8, wspace=0.1, aspect=1.3,
                gamma=0.8,
                panel = '(a)', panel_xy=(0.12, 1.), panel_fontsize=30,
                save_png=None, save_eps=None):
        if gamma!=1:
            data = gamma_correction(data, gamma)
        
        x = data.shape[1]
        y = data.shape[2]
        a = position[0]
        b = position[1]

        fig = plt.figure(figsize=figsize)
        spec = gridspec.GridSpec(ncols=n_shots, nrows=1,
                         wspace=wspace
                         )
        for i in range(n_shots):
            ax = fig.add_subplot(spec[i])
            if i==0:
                ax.set_title(title, loc='left')
            ax.imshow(np.rot90(data[start+i*step], -1), cmap='Greys_r', norm=Normalize(vmin=0, vmax=255), aspect=aspect)
            if scanline!=False:
                if i==0:
                    ax.plot([x-b, x-b], [a-int(width/2), a+int(width/2)], color=c_line, linewidth=lw_line)
            ax.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
            if i==0:
                fig.text(ax.get_position().x1-panel_xy[0], ax.get_position().y1-panel_xy[1], s=panel, fontsize=panel_fontsize)
            if start_t_is_0:
                ax.set_xlabel('{:.1f}'.format(i*step*(t_data[1]-t_data[0])*1000)+' [ms]')
            else:
                ax.set_xlabel('{:.1f}'.format(t_data[start+i*step]*1000)+' [ms]')
       
        plt.tight_layout()
        if save_png==None:
            plt.show()
        else:
            plt.savefig(save_png+'.png', bbox_inches="tight")
        if save_png==None:
            return
        else:
            plt.savefig(save_eps+'.eps', bbox_inches="tight")

    def figure04(self, 
                data, t, lyapunov_exponents, 
                figsize=(25, 5), width_ratios=[4, 7, 8], wspace=0.3,
                n_shift=25, n_initdel=2000, n_plt=3000, same_lim=True,
                freq_lim=(50, 350),
                n_dim=4, lyapunov_lim=(-110, 20), 
                title0=None, title1=None, title2=None, title_loc0='left', title_loc1='left', title_loc2='left', 
                xlabel0=r'$x(t)$', ylabel0=r'$x(t-\tau)$', linestyle0='-', c0='k', lw0=4,
                xlabel1='Frequency [Hz]', ylabel1='Power Spectral Density', linestyle1='-', c1='k', lw1=4,
                xlabel2='Dimension', ylabel2='Lyapunov Exponents', linestyle2_0line='dashed', c2_0line='k', lw2_0line=4,
                linestyle2='dashed', c2='k', lw2=4, marker='o', markersize=15,
                panel_list = ['(a)', '(b)', '(c)'], panel_xy_list=[(0.17, 1.05), (0.27, 1.05), (0.31, 1.05)], panel_fontsize=40, 
                save_png=None, save_eps=None):
        spec = gridspec.GridSpec(ncols=3, nrows=1, width_ratios=width_ratios, wspace=wspace)
        fig = plt.figure(figsize=figsize)
        freq_model, amp_model = fft(data, t)

        ax1 = fig.add_subplot(spec[0])
        ax1.set_title(title0, loc=title_loc0)
        ax1.plot(data[n_initdel:][n_shift:][:n_plt], data[n_initdel:][:-n_shift][:n_plt], linestyle=linestyle0, c=c0, lw=lw0)
        ax1.set_xlabel(xlabel0)
        ax1.set_ylabel(ylabel0)
        ax1.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax1.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax1.set_aspect('equal', 'datalim')
        fig.text(ax1.get_position().x1-panel_xy_list[0][0], ax1.get_position().y1-panel_xy_list[0][1], s=panel_list[0], fontsize=panel_fontsize)

        ax2 = fig.add_subplot(spec[1])
        ax2.set_title(title1, loc=title_loc1)
        ax2.plot(freq_model, amp_model, linestyle=linestyle1, lw=lw1, c=c1)
        ax2.set_xlabel(xlabel1)
        ax2.set_ylabel(ylabel1)
        ax2.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%d'))
        ax2.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax2.set_xlim(freq_lim)
        fig.text(ax2.get_position().x1-panel_xy_list[1][0], ax2.get_position().y1-panel_xy_list[1][1], s=panel_list[1], fontsize=panel_fontsize)

        ax3 = fig.add_subplot(spec[2])
        ax2.set_title(title2, loc=title_loc2)
        ax3.axhline(y=0, xmin=0, xmax=n_dim+1, linestyle=linestyle2_0line, c=c2_0line, lw=lw2_0line)
        ax3.plot(np.arange(1, n_dim+1), lyapunov_exponents[:n_dim], linestyle=linestyle2, c=c2, 
                lw=lw2, marker=marker, markersize=markersize)
        ax3.grid()
        ax3.set_ylim(lyapunov_lim)
        ax3.set_xlabel(xlabel2)
        ax3.set_ylabel(ylabel2)
        fig.text(ax3.get_position().x1-panel_xy_list[2][0], ax3.get_position().y1-panel_xy_list[2][1], s=panel_list[2], fontsize=panel_fontsize)

        plt.tight_layout()
        if save_png==None:
            plt.show()
        else:
            plt.savefig(save_png+'.png', bbox_inches="tight")
        if save_png==None:
            return
        else:
            plt.savefig(save_eps+'.eps', bbox_inches="tight")

    def figure05(self, 
                scanned_video, latent_vector, latent_vector_discreted, latent_vector_discreted_idx, t, bifurcation_data, bifurcation_params,
                conversion_param=40.,
                figsize=(25, 5), width_ratios=[1, 1], wspace=0.5, height_ratios=[1, 1], hspace=0.5,
                title1=None, title2=None, title3=None,
                aspect=0.9, gamma=0.3,
                lw_lv=3, c_lv='k', label_lv='Latent vector',
                marker_lv_lm='o', s_lv_lm=10, c_lv_lm='b', label_lv_lm='Local maxima', legend=False,
                xlim=(None, None), ylim=(None, None), xlabel=r'$mu $'+'[l/min]', ylabel=r'$y_l$', marker='.', c='b', s=50,
                panel_list = ['(a)', '(b)', '(c)'], panel_xy_list=[(0.17, 1.05), (0.27, 1.05), (0.31, 1.05)], panel_fontsize=40, 
                save_png=None, save_eps=None):
        spec = gridspec.GridSpec(ncols=2, nrows=2, width_ratios=width_ratios, wspace=wspace, height_ratios=height_ratios, hspace=hspace)
        fig = plt.figure(figsize=figsize)

        scanned_video = gamma_correction(scanned_video, gamma)

        ax1 = fig.add_subplot(spec[0, 0])
        ax1.set_title(title1, loc='left')
        ax1.imshow(scanned_video.T, cmap='Greys_r', norm=Normalize(vmin=0, vmax=255), aspect=aspect)
        ax1.tick_params(labelleft=False, left=False, labelbottom=False, bottom=True)
        fig.text(ax1.get_position().x1-panel_xy_list[0][0], ax1.get_position().y1-panel_xy_list[0][1], s=panel_list[0], fontsize=panel_fontsize)

        ax2 = fig.add_subplot(spec[1 ,0], xmargin=0)
        ax2.set_title(title2, loc='left')
        ax2.plot((t-t[0])*1000, latent_vector, lw=lw_lv, c=c_lv, label=label_lv)
        ax2.plot(((t-t[0])[latent_vector_discreted_idx])*1000, latent_vector_discreted, linestyle='None', lw=0, c=c_lv_lm, marker=marker_lv_lm, markersize=s_lv_lm, label=label_lv_lm)
        ax2.set_xlabel('Time [ms]')
        ax2.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%d'))
        ax2.tick_params(labelleft=False, left=False, labelbottom=True, bottom=True)
        if legend:
            ax2.legend(frameon=False)
        fig.text(ax2.get_position().x1-panel_xy_list[1][0], ax2.get_position().y1-panel_xy_list[1][1], s=panel_list[1], fontsize=panel_fontsize)

        ax3 = fig.add_subplot(spec[0:, 1])
        ax3.set_title(title3, loc='left')
        for i in range(len(bifurcation_data)):  
            ax3.scatter(np.full(bifurcation_data[i].shape, bifurcation_params[i]*conversion_param), bifurcation_data[i], marker=marker, c=c, s=s)
        ax3.tick_params(labelleft=False, left=False, labelbottom=True, bottom=True)
        ax3.set_xlim(xlim)
        ax3.set_ylim(ylim)
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(ylabel)
        ax3.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%d'))
        fig.text(ax3.get_position().x1-panel_xy_list[2][0], ax3.get_position().y1-panel_xy_list[2][1], s=panel_list[2], fontsize=panel_fontsize)

        plt.tight_layout()
        if save_png==None:
            plt.show()
        else:
            plt.savefig(save_png+'.png', bbox_inches="tight")
        if save_png==None:
            return
        else:
            plt.savefig(save_eps+'.eps', bbox_inches="tight")