# -*- coding: utf-8 -*-
"""
Created on Tue Aug 2 16:38:44 2022

author: Dr. Dmitrii Potorochin
email:  dmitrii.potorochin@desy.de
        dmitrii.potorochin@physik.tu-freiberg.de
        dm.potorochin@gmail.com
"""

# This section is supposed for importing necessary modules.
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import h5py
import json
import calendar
from types import SimpleNamespace
from time import gmtime
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
import matplotlib
from scipy.signal import savgol_filter
import xarray as xr
from PIL import Image, ImageDraw, ImageFont
from lmfit.models import VoigtModel, ConstantModel
import matplotlib.colors as colors

from timeit import default_timer as timer

# Dictionary for colors
color_dict = {
  0: 'blue',
  1: 'tab:red',
  2: 'black',
  3: 'tab:orange',
  4: 'tab:green',
  5: 'deeppink',
  6: 'tab:cyan',
  7: 'magenta',
  8: 'yellow'
}


def scan_hdf5(hdf5_obj, hdf5_path=[]):
    '''
    This function helps to adapt to changing structure
    of hdf5 files from WESPE.
    '''
    if type(hdf5_obj) in [h5py._hl.group.Group, h5py._hl.files.File]:
        for counter, key in enumerate(hdf5_obj.keys()):
            scan_hdf5(hdf5_obj[key])
    elif type(hdf5_obj) == h5py._hl.dataset.Dataset:
        full_path = hdf5_obj.name
        dataset_name = full_path.replace(hdf5_obj.parent.name, '')
        dataset_name = dataset_name.replace('/', '')
        if dataset_name == 'energy_Grid_ROI':
            hdf5_path.append(hdf5_obj.parent.name)
    return hdf5_path


def text_phantom(text, size):
    '''
    This function helps to create a dummy image with an error message
    if something goes wrong with data handling.
    '''
    # Availability is platform dependent
    font = 'arial'

    # Create font
    pil_font = ImageFont.truetype(font + ".ttf", size=size // (len(text)//4),
                                  encoding="unic")
    text_width, text_height = pil_font.getsize(text)

    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', [size, size//2], (0, 0, 0))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = ((size - text_width//2) // 2,
              (size//2 - text_height) // 2)
    white = "#FFFFFF"
    draw.text(offset, text, font=pil_font, fill=white)

    # Convert the canvas into an array with values in [0, 1]
    return (255 - np.asarray(canvas)) / 255.0


class create_batch:
    '''
    The object for storing data of combined runs.
    '''

    def __init__(self, file_dir, run_list, DLD='DLD4Q'):
        '''
        This initialization happens on 'Upload runs'.
        '''
        self.file_dir = file_dir
        self.batch_dir, self.batch_list = [], []
        for run_number in run_list:
            file_name = f'{run_number}' + os.sep + f'{run_number}_energy.mat'
            file_full = file_dir + os.sep + file_name
            self.batch_list.append(read_file(file_full, DLD=DLD))
            self.batch_dir.append(file_full)

        full_info = []
        for i in self.batch_list:
            full_info.append(i.info)
        self.full_info = 'DETAILED INFO:\n\n' + '\n\n'.join(full_info)

        title = 'SHORT SUMMARY:\n'
        run_num, is_static, KE, mono = [], [], [], []
        for i in self.batch_list:
            run_num.append(str(i.run_num))
            is_static.append(i.is_static)
            KE.append(i.KE)
            mono.append(i.mono_mean)
        # Run numbers
        self.run_num = ', '.join(run_num)
        run_num = [int(i) for i in run_num]
        run_num.sort()
        if len(run_num) == 1:
            run_num = f'Uploaded run: {run_num[0]}'
        elif len(run_num) > 6:
            run_num = f'Uploaded runs: {np.min(run_num)}-{np.max(run_num)}'
        else:
            run_num = [str(i) for i in run_num]
            run_num = ', '.join(run_num)
            run_num = 'Uploaded runs: ' + run_num
        self.run_num_o = run_num.replace('Uploaded runs: ', '')
        # Static scan check
        if all(is_static):
            is_static_s = 'Static check: All runs are static (+)'
        elif not any(is_static):
            is_static_s = 'Static check: All runs are delay scans (+)'
        else:
            is_static_s = 'Static check: Delay scans are mixed with static scans (!!!)'
        # Energy region check
        if np.max(KE) - np.min(KE) > 5:
            KE_s = 'Region check: Various energy regions are on the list (!!!)'
        else:
            KE_s = 'Region check: Homogeneous energy regions (+)'
        # Mono check
        if np.max(mono) - np.min(mono) > 0.15:
            mono_s = 'Mono check: Various mono values for different runs (!!!)'
        else:
            mono_s = 'Mono check: No mono energy jumps detected (+)'
        self.en_threshold = np.max(mono) + 50
        if self.en_threshold < 50:
            self.en_threshold = 1000
        static_cut_list = []
        for counter, i in enumerate(is_static):
            if i is True:
                static_cut = np.mean(self.batch_list[counter].DLD_delay)
                static_cut_list.append(static_cut)
        self.static_cut_list = static_cut_list
        short_info = [title, run_num, is_static_s, KE_s, mono_s]
        self.short_info = '\n'.join(short_info) + '\n\n'

    def time_zero(self, t0=1328.2):
        '''
        Method for creating new array coordinate 'Delay relative t0'
        after specification of the delay stage value considered as time zero.
        '''
        self.t0 = read_file.rounding(t0, self.delay_step)
        t0_coord = self.t0 - self.delay_energy_map.coords['Delay stage values']
        self.delay_energy_map.coords['Delay relative t0'] = t0_coord
        self.delay_energy_map.coords['Delay'] = t0_coord
        self.delay_energy_map.attrs['Time axis'] = 'Delay relative t0'

    def create_map(self):
        '''
        This method sums delay-energy maps of individual runs
        uploaded to the batch.
        '''
        self.energy_step = self.batch_list[0].energy_step
        self.delay_step = self.batch_list[0].delay_step
        self.ordinate = self.batch_list[0].ordinate
        attrs = self.batch_list[0].delay_energy_map.attrs
        for counter, i in enumerate(self.batch_list):
            if counter == 0:
                total_map = i.delay_energy_map
            else:
                total_map = total_map + i.delay_energy_map
        total_map.attrs = attrs
        try:
            total_map.coords['Binding energy']
            total_map.coords['Kinetic energy']
        except KeyError:
            total_map = xr.DataArray([])
        if np.min(total_map.values.shape) == 0:
            concat_list = []
            for counter, i in enumerate(self.batch_list):
                concat_list.append(i.delay_energy_map)
            total_map = xr.combine_by_coords(concat_list, compat='override')
            total_map.coords['Delay stage values'] = total_map.coords['Delay']
            total_map = total_map.to_array(dim='variable', name=None)
            total_map = total_map.sum(dim='variable')
            total_map.attrs = attrs

        if np.min(total_map.values.shape) == 0:
            total_map.attrs['Merge successful'] = False

        # Filter empty EDCs
        y_check = total_map.sum('Energy', skipna=True)
        y_check = y_check/total_map.coords['Energy'].shape[0]
        remove_list = np.where(y_check < 1)
        remove_list = y_check.coords['Delay'][remove_list]
        for i in remove_list:
            total_map = total_map.where(total_map['Delay'] != i, drop=True)

        shape = total_map.coords['Delay'].values.shape[0]
        total_map.coords['Delay index'] = ('Delay', np.arange(shape))
        self.delay_energy_map = total_map.fillna(0)
        self.delay_energy_map = self.delay_energy_map.where(self.delay_energy_map.coords['Kinetic energy'].notnull(), drop=True)
        if np.median(np.gradient(self.delay_energy_map.coords['Binding energy'].values)) > 0:
            self.delay_energy_map=self.delay_energy_map.isel(Energy=slice(None, None, -1))
        self.delay_energy_map_plot = self.delay_energy_map

    def create_dif_map(self):
        '''
        This method generates a difference map by averaging data before
        -0.25 ps and subtracting it from the delay-energy map.
        '''
        attrs = self.delay_energy_map_plot.attrs
        t_axis_step = self.delay_energy_map_plot.coords['Delay'].values
        try:
            t_axis_step = abs(np.median(np.gradient(t_axis_step)))
        except ValueError:
            t_axis_step = 1
        t_axis_step = int(-2.5*t_axis_step)
        norm = self.delay_energy_map_plot.loc[t_axis_step:].mean('Delay')
        self.delay_energy_map_dif = self.delay_energy_map_plot - norm
        self.delay_energy_map_dif.attrs = attrs

    def set_BE(self):
        '''
        Method for switching visualization to 'Binding energy'
        coordinate of 'Energy' dimension.
        '''
        coord = self.delay_energy_map_plot.coords['Binding energy']
        self.delay_energy_map_plot.coords['Energy'] = coord
        self.delay_energy_map_plot.attrs['Energy axis'] = 'Binding energy'

    def set_KE(self):
        '''
        Method for switching visualization to 'Kinetic energy'
        coordinate of 'Energy' dimension.
        '''
        coord = self.delay_energy_map_plot.coords['Kinetic energy']
        self.delay_energy_map_plot.coords['Energy'] = coord
        self.delay_energy_map_plot.attrs['Energy axis'] = 'Kinetic energy'

    def set_T0(self):
        '''
        Method for switching visualization to 'Delay relative t0'
        coordinate of 'Delay' dimension.
        '''
        coord = self.delay_energy_map_plot.coords['Delay relative t0']
        self.delay_energy_map_plot.coords['Delay'] = coord
        self.delay_energy_map_plot.attrs['Time axis'] = 'Delay relative t0'

    def set_Tds(self):
        '''
        Method for switching visualization to 'Delay stage values'
        coordinate of 'Delay' dimension.
        '''
        coord = self.delay_energy_map_plot.coords['Delay stage values']
        self.delay_energy_map_plot.coords['Delay'] = coord
        self.delay_energy_map_plot.attrs['Time axis'] = 'Delay stage values'

    def set_dif_map(self):
        '''
        Method for switching visualization to the difference plot.
        '''
        self.delay_energy_map_plot = self.delay_energy_map_dif

    def ROI(self, limits, axis, mod_map=True):
        '''
        Method for selecting the range of values of interest
        on delay-energy map.
        limits - a list for determination of a minimum and
        a maximum of the range
        axis - selection of Time or Energy axis
        mod_map - if True, changes the array used for visualization
        returns a new array after cutting out undesired regions
        '''
        min_val = np.min(limits)
        max_val = np.max(limits)
        if axis == 'Time axis':
            if self.delay_energy_map_plot.attrs[axis] == 'Delay stage values':
                new_a = self.delay_energy_map_plot.loc[min_val:max_val]
            else:
                new_a = self.delay_energy_map_plot.loc[max_val:min_val]
        if axis == 'Energy axis':
            if self.delay_energy_map_plot.attrs[axis] == 'Kinetic energy':
                new_a = self.delay_energy_map_plot.loc[:, min_val:max_val]
            else:
                new_a = self.delay_energy_map_plot.loc[:, max_val:min_val]
        if mod_map is True:
            self.delay_energy_map_plot = new_a
        else:
            return new_a

    def norm_total_e(self):
        '''
        Method for normalization of delay-energy map in terms of the concept
        that every time delay line should contain the same number of detected
        electrons, i.e., we have only redistribution of electrons in the
        energy domain.
        '''
        arr = self.delay_energy_map_plot
        attrs = self.delay_energy_map_plot.attrs

        norm = arr.sum('Energy', skipna=True)
        new_arr = arr/norm * norm.mean('Delay')

        self.delay_energy_map_plot = new_arr
        self.delay_energy_map_plot.attrs = attrs
        self.delay_energy_map_plot.attrs['Normalized'] = True

    def norm_01(self):
        '''
        Method for normalization of delay-energy map to zero to one intensity.
        '''
        arr = self.delay_energy_map_plot
        attrs = self.delay_energy_map_plot.attrs

        norm = arr.min('Energy', skipna=True)
        norm = norm.min('Delay', skipna=True)
        new_arr = arr - norm
        norm = new_arr.max('Energy', skipna=True)
        norm = norm.max('Delay', skipna=True)
        new_arr = new_arr/norm

        self.delay_energy_map_plot = new_arr
        self.delay_energy_map_plot.attrs = attrs
        self.delay_energy_map_plot.attrs['Normalized'] = True

    def norm_11(self):
        '''
        Method for normalization of delay-energy map to minus one to one
        intensity range. Either high or low limit absolute value is one.
        The other limit is scaled accordingly.
        It suits well for the difference plot.
        '''
        arr = self.delay_energy_map_plot
        attrs = self.delay_energy_map_plot.attrs

        pos_norm = arr.max('Energy', skipna=True)
        pos_norm = pos_norm.max('Delay', skipna=True)
        neg_norm = arr.min('Energy', skipna=True)
        neg_norm = neg_norm.min('Delay', skipna=True)
        norm = xr.concat((pos_norm, neg_norm), dim='New')
        norm = np.abs(norm)
        norm = norm.max('New', skipna=True)
        new_arr = arr/norm

        self.delay_energy_map_plot = new_arr
        self.delay_energy_map_plot.attrs = attrs
        self.delay_energy_map_plot.attrs['Normalized'] = True

    def t0_cut(self, position='Main', hv=2.407, axis='Energy axis'):
        '''
        Method for simplification of finding the position of the most
        prominent feature or sidebands for the t0 finder.
        '''
        position_list = str(position)
        position_list = position_list.split(',')
        position_list = [i.strip().lower() for i in position_list]
        position = position_list[0]

        array = self.delay_energy_map_plot
        if axis == 'Energy axis':
            array_mean = array.median('Delay')
            pos_e = array_mean.idxmax('Energy').values
        else:
            array_mean = array.median('Energy')
            pos_e = array_mean.idxmax('Delay').values

        if position == 'sb':
            try:
                hv = position_list[1]
                hv = float(hv)
                pos_e += hv
            except IndexError:
                pos_e += hv

        try:
            pos_e = float(position)
        except ValueError:
            pass

        pos_e = np.around(pos_e, 2)
        return pos_e

    def save_map_dat(self):
        '''
        Method for saving the delay-energy map from visualization
        to ASCII format.
        One can find the saved result in the 'ASCII_output' folder.
        '''
        arr = self.delay_energy_map_plot
        length = arr.shape[0]
        ts = calendar.timegm(gmtime())
        date_time = datetime.fromtimestamp(ts)
        str_date_time = date_time.strftime("%d.%m.%Y_%H-%M-%S")
        path = self.file_dir + os.sep + 'ASCII_output'
        if os.path.isdir(path) is False:
            os.mkdir(path)
        path = path + os.sep + 'Maps'
        if os.path.isdir(path) is False:
            os.mkdir(path)
        path = path + os.sep + str_date_time + os.sep
        if os.path.isdir(path) is False:
            os.mkdir(path)
        with open(path+"Summary.txt", "w") as text_file:
            text_file.write(f'Loaded runs: {self.run_num_o}\n')
            text_file.write(f'Energy step: {self.energy_step} eV\n')
            if self.ordinate == 'delay':
                text_file.write(f'Delay step: {self.delay_step} ps\n')
            else:
                text_file.write(f'MicroBunch step: {self.delay_step} u.\n')
            if arr.attrs['Energy axis'] == 'Kinetic energy':
                text_file.write('Energy axis: Kinetic energy (column 1)\n')
            else:
                text_file.write('Energy axis: Binding energy (column 1)\n')
            if self.ordinate == 'delay':
                if arr.attrs['Time axis'] == 'Delay relative t0':
                    text_file.write('Time axis: Delay relative t0 (file name)\n')
                else:
                    text_file.write('Time axis: Delay stage values (file name)\n')
            elif self.ordinate == 'MB_ID':
                text_file.write('Time axis: MicroBunch ID (file name)\n')
            if arr.attrs['Normalized'] is True:
                text_file.write('Normalized: True\n')
            else:
                text_file.write('Normalized: False\n')
        for i in range(length):
            x = arr.coords['Energy'].values
            x = list(x)
            x = [read_file.rounding(i, self.energy_step) for i in x]
            x = np.array(x)
            x = np.expand_dims(x, axis=0)
            y = arr.isel(Delay=i).values
            y = np.expand_dims(y, axis=0)
            out = np.append(x, y, axis=0)
            out = np.rot90(out)

            file_full = path
            delay_val = arr.coords['Delay'].values[i]
            delay_val = np.around(delay_val, 2)
            if arr.attrs['Time axis'] == 'Delay relative t0':
                order = length - 1 - i
            else:
                order = i
            if len(str(order)) == len(str(length)):
                order = str(order)
            else:
                order = str(order)
                for j in range(len(str(length))-len(str(order))):
                    order = '0' + order
            if self.ordinate == 'delay':
                file_full = file_full + f'{order}_{delay_val} ps.dat'
            else:
                file_full = file_full + f'{order}_{delay_val} u.dat'
            np.savetxt(file_full, out, delimiter='    ')
            print(f"Saved as {file_full}")

    def axs_plot(self, axs):
        # Loading configs from json file.
        try:
            with open('config.json', 'r') as json_file:
                config = json.load(json_file)
        except FileNotFoundError:
            with open('packages/config.json', 'r') as json_file:
                config = json.load(json_file)
        config = json.dumps(config)
        config = json.loads(config,
                            object_hook=lambda d: SimpleNamespace(**d))
        '''
        Method for creating matplotlib axes for delay-energy map visualization.
        '''
        if np.min(self.delay_energy_map_plot.values.shape) == 0:
            if self.delay_energy_map_plot.attrs['Merge successful'] is False:
                label = ['Merge was not successful. You can:',
                         '1) Change delay/energy step;',
                         '2) Switch to different axis (e.g., KE/BE).']
                label = '\n'.join(label)
                im1 = axs.imshow(text_phantom(label, 1000))
                axs.axis('off')
            else:
                label = ['The delay-energy map is empty!',
                         '1) Check ROI values;',
                         '2) Check bunch filtering.']
                label = '\n'.join(label)
                im1 = axs.imshow(text_phantom(label, 1000))
                axs.axis('off')
        else:
            image_data = self.delay_energy_map_plot.values
            image_data_y = self.delay_energy_map_plot.coords['Delay'].values
            image_data_x = self.delay_energy_map_plot.coords['Energy'].values
            if image_data.shape[0] == 1:
                image_data = np.pad(image_data, [(1, 1), (0, 0)],
                                    mode='constant')
                image_data_y = [image_data_y[0]-1,
                                image_data_y[0],
                                image_data_y[0]+1]
                image_data_y = np.array(image_data_y)
            self.varied_y_step = False
            if image_data_y.shape[0] > 1:
                if np.around(np.std(np.gradient(image_data_y)), 3) > 0:
                    self.varied_y_step = True
                    image_data_y = np.arange(image_data_y.shape[0])
                    self.image_data_y = image_data_y
                    if self.delay_energy_map_plot.attrs['Time axis'] == 'Delay relative t0':
                        pos_list = np.linspace(np.min(image_data_y),
                                               np.max(image_data_y),
                                               config.map_n_ticks_y,
                                               dtype=int)
                    else:
                        pos_list = np.linspace(np.max(image_data_y),
                                               np.min(image_data_y),
                                               config.map_n_ticks_y,
                                               dtype=int)
                    label_list = self.delay_energy_map_plot.coords['Delay']
                    label_list = label_list[pos_list].values
            self.map_z_max = np.nanmax(image_data)
            self.map_z_min = np.nanmin(image_data)
            self.map_z_tick = (self.map_z_max - self.map_z_min)/config.map_n_ticks_z
            if self.map_z_tick < 1:
                self.map_z_tick_decimal = 1
            else:
                self.map_z_tick_decimal = 0
            self.map_z_tick = round(self.map_z_tick, self.map_z_tick_decimal)
            if self.map_z_tick == 0:
                self.map_z_tick = 1

            self.map_y_max = np.nanmax(image_data_y)
            self.map_y_min = np.nanmin(image_data_y)
            self.map_y_tick = (self.map_y_max - self.map_y_min)/config.map_n_ticks_y
            if self.map_y_tick < 1:
                self.map_y_tick_decimal = 1
            else:
                self.map_y_tick_decimal = 0
            self.map_y_tick = round(self.map_y_tick, self.map_y_tick_decimal)
            if self.map_y_tick == 0:
                self.map_y_tick = 1

            self.map_x_max = np.nanmax(image_data_x)
            self.map_x_min = np.nanmin(image_data_x)
            self.map_x_tick = (self.map_x_max - self.map_x_min)/config.map_n_ticks_x
            self.map_x_tick = math.ceil(self.map_x_tick)
            if self.map_x_tick == 0:
                self.map_x_tick = 1

            if self.delay_energy_map_plot.attrs['Energy axis'] == 'Kinetic energy':
                x_start = np.min(image_data_x)
                x_end = np.max(image_data_x)
            else:
                x_start = np.max(image_data_x)
                x_end = np.min(image_data_x)

            if self.delay_energy_map_plot.attrs['Time axis'] == 'Delay stage values':
                y_start = np.max(image_data_y)
                y_end = np.min(image_data_y)
            elif self.varied_y_step is True:
                y_start = np.max(image_data_y)
                y_end = np.min(image_data_y)
            else:
                y_start = np.min(image_data_y)
                y_end = np.max(image_data_y)
                
            extent = [x_start, x_end,
                      y_start, y_end]

            vmin = np.min(image_data)
            vmax = np.max(image_data)*config.map_scale
            self.map_z_tick = self.map_z_tick*config.map_scale
            if vmin < 0:
                vmin = vmin*config.map_scale

            TwoSlopeNorm = config.TwoSlopeNorm
            if TwoSlopeNorm < 1 and TwoSlopeNorm > 0:
                im1 = axs.imshow(image_data, origin='upper',
                                 extent=extent,
                                 cmap=config.cmap, aspect='auto',
                                 norm = colors.TwoSlopeNorm(vmin=vmin,
                                                            vcenter=TwoSlopeNorm*vmax,
                                                            vmax=vmax))
            else:
                im1 = axs.imshow(image_data, origin='upper',
                                 extent=extent,
                                 vmin=vmin,
                                 vmax=vmax,
                                 cmap=config.cmap, aspect='auto')

            divider1 = make_axes_locatable(axs)
            cax1 = divider1.append_axes("right", size="3.5%", pad=0.09)
            cbar = plt.colorbar(im1, cax=cax1,
                                ticks=MultipleLocator(self.map_z_tick))
            cbar.minorticks_on()
            if self.delay_energy_map_plot.attrs['Normalized'] is True:
                cax1.set_ylabel('Intensity (arb. units)', rotation=270,
                                labelpad=30,
                                fontsize=config.font_size_axis*0.8)
            else:
                cax1.set_ylabel('Intensity (counts)', rotation=270,
                                labelpad=30,
                                fontsize=config.font_size_axis*0.8)

            run_list_s = self.run_num.split(', ')
            run_list = [int(i) for i in run_list_s]
            run_list.sort()
            if len(run_list) == 1:
                run_string = f'Run {run_list[0]}'
            elif len(run_list) > 4:
                run_string = f'Runs {np.min(run_list)}-{np.max(run_list)}'
            else:
                run_list_s = [str(i) for i in run_list_s]
                run_string = ', '.join(run_list_s)
                run_string = 'Runs ' + run_string
            axs.set_title(run_string, pad=15,
                          fontsize=config.font_size_axis*1.2,
                          fontweight="light")
            if self.delay_energy_map_plot.attrs['Energy axis'] == 'Binding energy':
                axs.set_xlabel('Binding energy (eV)', labelpad=5,
                               fontsize=config.font_size_axis)
            else:
                axs.set_xlabel('Kinetic energy (eV)', labelpad=5,
                               fontsize=config.font_size_axis)

            if self.ordinate == 'delay':
                axs.set_ylabel('Delay (ps)', labelpad=10,
                               fontsize=config.font_size_axis*0.8)
            elif self.ordinate == 'MB_ID':
                axs.set_ylabel('MicroBunch ID (units)', labelpad=10,
                               fontsize=config.font_size_axis*0.8)

            if self.delay_energy_map_plot.attrs['Time axis'] == 'Delay relative t0':
                position = 0
                if self.varied_y_step is True:
                    coord = self.delay_energy_map_plot.coords['Delay']
                    position = coord.sel(Delay=position, method="nearest")
                    position = coord.where(coord == position, drop=True)
                    position = position['Delay index'].values
                axs.axhline(y=position, color=config.color_t0_line,
                            linewidth=config.line_width_t0_line,
                            alpha=config.line_op_t0_line/100,
                            linestyle=config.line_type_t0_line)

            # y axis
            if self.varied_y_step is True:
                decimals = read_file.decimal_n(self.map_y_tick)
                label_list = [round(i, decimals) for i in label_list]
                axs.set_yticks(pos_list, label_list)
            else:
                axs.yaxis.set_major_locator(MultipleLocator(self.map_y_tick))
                axs.yaxis.set_minor_locator(MultipleLocator(self.map_y_tick /
                                                            config.map_n_ticks_minor))
            # x axis
            axs.xaxis.set_major_locator(MultipleLocator(self.map_x_tick))
            axs.xaxis.set_minor_locator(MultipleLocator(self.map_x_tick /
                                                        config.map_n_ticks_minor))
            axs.tick_params(axis='both', which='major',
                            length=config.map_tick_length,
                            width=config.map_tick_length/4)
            axs.tick_params(axis='both', which='minor',
                            length=config.map_tick_length/1.5,
                            width=config.map_tick_length/4)
            cax1.tick_params(axis='both', which='major',
                             length=config.map_tick_length,
                             width=config.map_tick_length/4)
            cax1.tick_params(axis='both', which='minor',
                             length=config.map_tick_length/1.5,
                             width=config.map_tick_length/4)
            if self.map_y_min == self.map_y_max:
                axs.set_ylim(self.map_y_min-1, self.map_y_max+1)
            if self.map_x_min == self.map_x_max:
                axs.set_xlim(self.map_x_min-1, self.map_x_max+1)


class read_file:
    '''
    The object for storing data from individual hdf5 files.
    It is used further for creating create_batch objects.
    '''

    def __init__(self, file_full, DLD='DLD4Q'):
        '''
        Object initialization where reading out of data from hdf5 files occurs.
        '''
        f = h5py.File(file_full, 'r')
        self.is_static = False
        self.file_full = file_full
        self.file_folder = file_full.split(os.sep)[:-1]
        self.file_folder = f'{os.sep}'.join(self.file_folder)
        self.run_num = file_full.split(os.sep)[-1].replace('_energy.mat', '')
        self.run_num = str(self.run_num)
        self.static = int(self.run_num)
        self.DLD = DLD

        hdf5_path_read = scan_hdf5(f)
        if len(hdf5_path_read) > 1:
            for path_i in hdf5_path_read:
                if DLD in path_i:
                    self.hdf5_path = path_i
                    break
                else:
                    self.hdf5_path = hdf5_path_read[0]
        else:
            self.hdf5_path = hdf5_path_read[0]

        self.DLD_energy = f.get(f'{self.hdf5_path}/energy_Grid_ROI')[0]
        self.e_num = self.DLD_energy.shape[0]
        self.BAM = f.get(f'{self.hdf5_path}/BAM')[0]
        try:
            self.GMD = f.get(f'{self.hdf5_path}/GMDBDA_Electrons')[0]
        except TypeError:
            self.GMD = 0
        try:
            self.mono = f.get(f'{self.hdf5_path}/mono')[0]
        except TypeError:
            self.mono = 0
        self.B_ID = f.get(f'{self.hdf5_path}/bunchID')[0]
        self.MB_ID = f.get(f'{self.hdf5_path}/microbunchID')[0]
        try:
            self.diode = f.get(f'{self.hdf5_path}/Pulse_Energy_DiodeBB')[0]
        except TypeError:
            self.diode = 0
        try:
            self.KE = f.get(f'param_backconvert_GUI/kinenergie_{self.DLD[-2:]}')
            self.KE = int(self.KE[0, 0])
        except TypeError:
            self.KE = f.get('param_backconvert_GUI/kinenergie')
            self.KE = int(self.KE[0])
        try:
            self.PE = f.get(f'param_backconvert_GUI/passenergie_{self.DLD[-2:]}')
            self.PE = int(self.PE[0, 0])
        except TypeError:
            self.PE = f.get('param_backconvert_GUI/passenergie')
            self.PE = int(self.PE[0])
        try:
            self.DLD_delay = f.get(f'{self.hdf5_path}/delay')[0]
        except TypeError:
            self.DLD_delay = np.full(self.DLD_energy.shape, self.static)
            self.is_static = True
        f.close()

        self.info = []
        self.info.append(f'File name: {file_full.split(os.sep)[-1]} / Electrons detected: {self.e_num}')
        self.info.append(f'Detector: {self.DLD} / KE: {self.KE} eV / PE: {self.PE} eV / Static: {str(self.is_static)}')
        
        self.B_num = int(np.max(self.B_ID) - np.min(self.B_ID))
        self.MB_num = int(np.max(self.MB_ID))
        self.mono_mean = np.around(np.mean(self.mono), 2)
        self.info.append(f'FEL mono: {self.mono_mean} eV / MacroBunches: {self.B_num} / MicroBunches: {self.MB_num}')

        self.KE_min = np.around(np.min(self.DLD_energy), 2)
        self.KE_max = np.around(np.max(self.DLD_energy), 2)
        self.KE_mean = np.around(np.mean(self.DLD_energy), 2)
        self.info.append(f'Min KE: {self.KE_min} eV / Max KE: {self.KE_max} eV / Mean KE: {self.KE_mean} eV')

        self.BE_min = np.around(self.mono_mean - self.KE_max - 4.5, 2)
        self.BE_max = np.around(self.mono_mean - self.KE_min - 4.5, 2)
        self.BE_mean = np.around(self.mono_mean - self.KE_mean - 4.5, 2)
        self.info.append(f'Min BE: {self.BE_min} eV / Max BE: {self.BE_max} eV / Mean BE: {self.BE_mean} eV')

        self.delay_min = np.around(np.min(self.DLD_delay), 2)
        self.delay_max = np.around(np.max(self.DLD_delay), 2)
        self.delay_mean = np.around(np.mean(self.DLD_delay), 2)
        self.info.append(f'Min delay: {self.delay_min} ps / Max delay: {self.delay_max} ps / Mean delay: {self.delay_mean} ps')

        self.GMD_min = np.around(np.min(self.GMD), 2)
        self.GMD_max = np.around(np.max(self.GMD), 2)
        self.GMD_mean = np.around(np.mean(self.GMD), 2)
        self.info.append(f'Min GMD: {self.GMD_min} / Max GMD: {self.GMD_max} / Mean GMD: {self.GMD_mean}')
        self.info = '\n'.join(self.info)

        self.B_ID_const = self.B_ID
        self.B_filter = False
        self.Macro_B_filter = 'All_Macro_B'
        self.Micro_B_filter = 'All_Micro_B'

    def Bunch_filter(self, B_range, B_type='MacroBunch'):
        '''
        Method for bunch filtering.
        B_range - a list for the determination of minimum and maximum values
        for bunch range of interest
            in percent for macrobunches
            in units for microbunches
        B_type - allows to select between 'MacroBunch' and 'MicroBunch'
        '''
        self.B_filter = True
        if B_type == 'MacroBunch':
            self.B_num = np.max(self.B_ID_const) - np.min(self.B_ID_const)
            B_min = np.min(self.B_ID_const)+(self.B_num)*min(B_range)/100
            B_max = np.min(self.B_ID_const)+(self.B_num)*max(B_range)/100
            min_list = np.where(self.B_ID < B_min)
            max_list = np.where(self.B_ID > B_max)
            print('Result of MacroBunch filtering:')
            self.Macro_B_filter = f'{int(B_min)}-{int(B_max)}_Macro_B'
        elif B_type == 'MicroBunch':
            B_min = min(B_range)
            B_max = max(B_range)
            min_list = np.where(self.MB_ID < B_min)
            max_list = np.where(self.MB_ID > B_max)
            print('Result of MicroBunch filtering:')
            self.Micro_B_filter = f'{int(B_min)}-{int(B_max)}_Micro_B'
        del_list = np.append(min_list, max_list)
        print(f'{len(del_list)} electrons removed from Run {self.run_num}')
        if del_list.size != 0:
            self.DLD_energy = np.delete(self.DLD_energy, del_list)
            self.DLD_delay = np.delete(self.DLD_delay, del_list)
            self.BAM = np.delete(self.BAM, del_list)
            if isinstance(self.GMD, int) is False:
                self.GMD = np.delete(self.GMD, del_list)
            if isinstance(self.mono, int) is False:
                self.mono = np.delete(self.mono, del_list)
            self.B_ID = np.delete(self.B_ID, del_list)
            self.MB_ID = np.delete(self.MB_ID, del_list)
            self.diode = np.delete(self.diode, del_list)

    def create_map(self, energy_step=0.05, delay_step=0.1,
                   ordinate='delay', save=True):
        self.ordinate = ordinate
        '''
        Method for creation of delay-energy map from the data loaded at
        initialization.
        energy_step and delay_step determine the bin size for
        'Energy' and 'Delay dimensions'
        '''
        self.energy_step = energy_step
        self.delay_step = delay_step
        save_path = self.file_folder + os.sep + 'netCDF_maps'
        if os.path.isdir(save_path) is False:
            os.mkdir(save_path)
        save_path = save_path + os.sep + f"{self.DLD}_{energy_step}eV"
        save_path = save_path + f"_{delay_step}ps"
        save_path = save_path + f"_{self.Macro_B_filter}"
        save_path = save_path + f"_{self.Micro_B_filter}.nc"
        try:
            if self.ordinate != 'delay':
                raise FileNotFoundError

            if save != 'on':
                raise FileNotFoundError

            with xr.open_dataset(save_path) as ds:
                loaded_map = ds

            for i in list(loaded_map.variables.keys()):
                if 'Run' in i:
                    data_name = i

            image_data = loaded_map.variables[data_name].values
            image_data_y = loaded_map.variables['Delay stage values'].values
            image_data_x = loaded_map.variables['Kinetic energy'].values
            coords = {"Delay stage values": ("Delay", image_data_y),
                      "Kinetic energy": ("Energy", image_data_x)}
            delay_energy_map = xr.DataArray(np.array(image_data),
                                            dims=["Delay", "Energy"],
                                            coords=coords)
            delay_energy_map.name = data_name
            BE = loaded_map.variables['Binding energy'].values
            delay_energy_map.coords['Binding energy'] = ('Energy', BE)

            delay_energy_map.coords['Energy'] = delay_energy_map.coords['Kinetic energy']
            delay_energy_map.coords['Delay'] = delay_energy_map.coords['Delay stage values']

            delay_energy_map.attrs = {'Delay units': 'ps',
                                      'Energy units': 'eV',
                                      'Time axis': 'Delay stage values',
                                      'Energy axis': 'Kinetic energy',
                                      'Normalized': False,
                                      'Type': 'Map',
                                      'Merge successful': True}
            self.delay_energy_map = delay_energy_map
            self.delay_energy_map_plot = self.delay_energy_map
        except FileNotFoundError:
            start = timer()
            '''
            This part is supposed to filter artifact values
            in the energy domain.
            '''
            for j in ['DLD_energy', 'DLD_delay']:
                i = getattr(self, j)
                mean = np.mean(i)
                std = np.std(i)
                min_list = np.where(i < mean-3*std)
                max_list = np.where(i > mean+3*std)
                del_list = np.append(min_list, max_list)
                if del_list.size != 0:
                    self.DLD_energy = np.delete(self.DLD_energy, del_list)
                    self.DLD_delay = np.delete(self.DLD_delay, del_list)
                    self.BAM = np.delete(self.BAM, del_list)
                    if isinstance(self.GMD, int) is False:
                        self.GMD = np.delete(self.GMD, del_list)
                    if isinstance(self.mono, int) is False:
                        self.mono = np.delete(self.mono, del_list)
                    self.B_ID = np.delete(self.B_ID, del_list)
                    self.MB_ID = np.delete(self.MB_ID, del_list)
                    if isinstance(self.diode, int) is False:
                        self.diode = np.delete(self.diode, del_list)
            '''
            Picking Delay or MB_ID as the ordinate axis.
            '''
            if ordinate == 'delay':
                parameter = self.DLD_delay
            elif ordinate == 'MB_ID':
                parameter = self.MB_ID

            DLD_delay_r = self.rounding(parameter, delay_step)
            DLD_energy_r = self.rounding(self.DLD_energy, energy_step)
            DLD_delay_r = np.around(DLD_delay_r,
                                    self.decimal_n(delay_step))
            DLD_energy_r = np.around(DLD_energy_r,
                                     self.decimal_n(energy_step))

            if config.map_counting == 'classic':
                '''
                Here we create a dictionary (delay_info), which stores all
                delay values as keys.
                Then, we assign delays to the numbers of count events.
                '''
                delay_info = {}
                DLD_delay_mean = DLD_delay_r.mean()
                DLD_delay_std = DLD_delay_r.std()
                if DLD_delay_std == 0:
                    DLD_delay_std = 1
                for counter, i in enumerate(DLD_delay_r):
                    if abs(i-DLD_delay_mean) < DLD_delay_std*10:
                        if i not in delay_info.keys():
                            delay_info[i] = [counter]
                        else:
                            delay_info[i].append(counter)
                '''
                Further, we sort the dictionary in ascending order of
                key values (delay values).
                '''
                delay_info_sorted = {}
                for i in sorted(delay_info.keys()):
                    delay_info_sorted[i] = delay_info[i]

                '''
                Here we create the main database for the whole file:
                    delay_energy_data[i][j][k]
                    i is responsible for the time axis, within every element we
                    have three cells, [j = 0, 1, 2] the first cell stores delay
                    value, the second cell stores a list containing kinetic
                    energies,the third cell contains the number of counts detected.
                '''
                delay_energy_data = []
                for i in delay_info_sorted.keys():
                    energy_info = {}
                    for event in delay_info_sorted[i]:
                        if DLD_energy_r[event] not in energy_info.keys():
                            energy_info[DLD_energy_r[event]] = 1
                        else:
                            energy_info[DLD_energy_r[event]] += 1
                    for eng in np.arange(min(DLD_energy_r), max(DLD_energy_r)
                                         + energy_step, energy_step):
                        if np.around(eng, decimals=
                                     self.decimal_n(energy_step)) not in energy_info.keys():
                            energy_info[np.around(eng, decimals=
                                                  self.decimal_n(energy_step))] = 0
                    energy_list = sorted(energy_info.keys())  # Sorting of KE

                    intensity_list = []
                    for energy in energy_list:  # Sorting of Counts along KE
                        intensity_list.append(energy_info[energy])
                    delay_energy_data.append([i, energy_list, intensity_list])
                    # Every cycle creates a cell [delay, [KE], [Counts]]

                '''

                CREATING ENERGY-DELAY MAP

                image_data - the map itself
                image_data_x - dictionary, where keys correspond to energy values,
                values correspond to the number of pixels along the x-axis
                image_data_y - dictionary, where keys() correspond to delay,
                values() correspond to the number of pixels along the y-axis

                '''
                image_data = []
                for i in delay_energy_data:
                    image_data.append(i[2])

                image_data_x = []
                for i in delay_energy_data[0][1]:
                    image_data_x.append(i)

                image_data_y = []
                for i in delay_energy_data:
                    image_data_y.append(i[0])

            else:
                image_data_x = np.arange(DLD_energy_r.min(),
                                         DLD_energy_r.max()+energy_step,
                                         energy_step)
                image_data_x = np.around(image_data_x,
                                         self.decimal_n(energy_step))
                image_data_y = np.arange(DLD_delay_r.min(),
                                         DLD_delay_r.max()+delay_step,
                                         delay_step)
                image_data_y = np.around(image_data_y,
                                         self.decimal_n(delay_step))

                image_data = []
                for i in image_data_y:
                    array_1 = DLD_energy_r[np.where(DLD_delay_r == i)]
                    line = []
                    array_1 = array_1.astype('f')
                    for j in image_data_x:
                        array_2 = np.where(array_1 == j)[0]
                        line.append(array_2.shape[0])
                    image_data.append(line)

            coords = {"Delay stage values": ("Delay", image_data_y),
                      "Kinetic energy": ("Energy", image_data_x)}
            delay_energy_map = xr.DataArray(np.array(image_data),
                                            dims=["Delay", "Energy"],
                                            coords=coords)
            delay_energy_map.name = 'Run ' + str(self.run_num)
            BE = self.mono_mean - np.array(image_data_x) - 4.5
            BE = np.around(self.rounding(BE, energy_step), self.decimal_n(energy_step))
            delay_energy_map.coords['Binding energy'] = ('Energy', BE)

            delay_energy_map.coords['Energy'] = delay_energy_map.coords['Kinetic energy']
            delay_energy_map.coords['Delay'] = delay_energy_map.coords['Delay stage values']

            if self.ordinate == 'delay':
                units = 'ps'
            elif self.ordinate == 'MB_ID':
                units = 'u.'
            delay_energy_map.attrs = {'Delay units': f'{units}',
                                      'Energy units': 'eV',
                                      'Time axis': 'Delay stage values',
                                      'Energy axis': 'Kinetic energy',
                                      'Normalized': False,
                                      'Type': 'Map',
                                      'Merge successful': True}

            self.delay_energy_map = delay_energy_map
            self.delay_energy_map_plot = self.delay_energy_map
            if save == 'on' and self.ordinate == 'delay':
                self.delay_energy_map.to_netcdf(save_path)
                print('Delay-energy map saved as:')
                print(save_path)
            end = timer()
            print(f'Run {self.run_num} done')
            print(f'Elapsed time: {round(end-start, 1)} s')

    def time_zero(self, t0=1328.2):
        '''
        Method for creating new array coordinate 'Delay relative t0'
        after specification of the delay stage value considered as time zero.
        '''
        self.t0 = self.rounding(t0, self.delay_step)
        t0_coord = self.t0 - self.delay_energy_map.coords['Delay stage values']
        self.delay_energy_map.coords['Delay relative t0'] = t0_coord
        self.delay_energy_map.coords['Delay'] = t0_coord
        self.delay_energy_map.attrs['Time axis'] = 'Delay relative t0'

    def create_dif_map(self):
        '''
        This method generates a difference map by averaging data before
        -0.25 ps and subtracting it from the delay-energy map.
        '''
        attrs = self.delay_energy_map_plot.attrs
        t_axis_step = self.delay_energy_map_plot.coords['Delay'].values
        try:
            t_axis_step = abs(np.median(np.gradient(t_axis_step)))
        except ValueError:
            t_axis_step = 1
        t_axis_step = int(-2.5*t_axis_step)
        norm = self.delay_energy_map.loc[t_axis_step:].mean('Delay')
        self.delay_energy_map_dif = self.delay_energy_map_plot - norm
        self.delay_energy_map_dif.attrs = attrs

    def set_BE(self):
        '''
        Method for switching xarray to 'Binding energy'
        coordinate of 'Energy' dimension.
        '''
        coord = self.delay_energy_map.coords['Binding energy']
        self.delay_energy_map.coords['Energy'] = coord
        self.delay_energy_map.attrs['Energy axis'] = 'Binding energy'

    def axs_plot(self, axs):
        # Loading configs from json file.
        try:
            with open('config.json', 'r') as json_file:
                config = json.load(json_file)
        except FileNotFoundError:
            with open('packages/config.json', 'r') as json_file:
                config = json.load(json_file)
        config = json.dumps(config)
        config = json.loads(config,
                            object_hook=lambda d: SimpleNamespace(**d))
        '''
        Method for creating matplotlib axes for delay-energy map visualization.
        Uses the corresponding method from the create_batch object.
        '''
        create_batch.axs_plot(self, axs)

    @staticmethod
    def rounding(x, y):
        '''
        The function rounds energy and delay values to the closest
        values separated by the desired step.
        x - input value
        y - desired step
        '''
        result = np.floor(x/y)*y
        check = (x / y) - np.floor(x/y)
        result = result + (check >= 0.5)*y
        return result

    @staticmethod
    def decimal_n(x):
        '''
        Determines the number of decimal points.
        '''
        result = len(str(x)) - 2
        if isinstance(x, int):
            result = 0
        return result


class map_cut:
    '''
    The object for creating and storing slices of the delay-energy map.
    '''

    def __init__(self, obj, positions, deltas,
                 axis='Time axis', approach='mean'):
        '''
        Initialization of the object when the creation of
        the slices is performed
        obj - create_batch object
        positions - a list of values where one wants to make a slice
        deltas - a list of slice widths
        axis - 'Time axis' or 'Energy axis' of the delay-energy map
        approach - 'mean' or 'sum' of individual lines within a slice
        '''
        self.run_num_o = obj.run_num_o
        self.energy_step = obj.energy_step
        self.delay_step = obj.delay_step
        self.file_dir = obj.file_dir
        self.ordinate = obj.ordinate
        self.plot_dif = False
        self.plot_dif = False
        self.delay_energy_map_plot = obj.delay_energy_map_plot
        try:
            self.varied_y_step = obj.varied_y_step
        except AttributeError:
            pass
        if isinstance(positions, list) is False:
            positions = [positions]
        if isinstance(deltas, list) is False:
            deltas = [deltas]

        if len(deltas) < len(positions):  # filling in missing delta values
            for i in range(len(positions)):
                try:
                    deltas[i]
                except IndexError:
                    try:
                        deltas.append(deltas[i-1])
                    except IndexError:
                        deltas.append(0.5)
        self.axis = axis
        self.arb_u = False
        self.positions = []
        self.deltas = []
        self.cuts = []
        self.map_show = []
        self.fit = False
        for counter, position in enumerate(positions):
            self.positions.append(position)
            self.deltas.append(deltas[counter])
            limit_1 = position - deltas[counter]/2
            limit_2 = position + deltas[counter]/2
            limits = [limit_1, limit_2]
            cut = obj.ROI(limits, axis, mod_map=False)
            self.e_axis = cut.attrs['Energy axis']
            self.t_axis = cut.attrs['Time axis']
            if axis == 'Time axis':
                self.coords = cut.coords['Energy'].values
                self.units = cut.attrs['Delay units']
                if approach == 'sum':
                    cut = cut.sum('Delay')
                else:
                    cut = cut.mean('Delay')
            else:
                self.coords = cut.coords['Delay'].values
                self.units = cut.attrs['Energy units']
                if approach == 'sum':
                    cut = cut.sum('Energy')
                else:
                    cut = cut.mean('Energy')
            if cut.isnull().values.all():
                dummy = np.zeros(self.coords.shape)
                self.cuts.append(dummy)
                self.map_show.append(False)
            else:
                self.cuts.append(cut.values)
                self.map_show.append(True)

        if self.units == 'ps':
            self.units_r = 'eV'
            self.var_n = 'T'
            self.var_n_r = 'E'
        elif self.units == 'u.':
            self.units_r = 'eV'
            self.var_n = 'MB'
            self.var_n_r = 'E'
        else:
            self.units_r = 'ps'
            self.var_n = 'E'
            self.var_n_r = 'T'

    def voigt_fit(self):
        '''
        Method for fitting of the very first slice with singular Voigt curve.
        It is supposed to be used for finding time zero.
        '''
        e_axis_step = np.gradient(self.delay_energy_map_plot.coords['Energy'].values).mean()
        x = self.coords
        y = self.cuts[0]

        model = VoigtModel() + ConstantModel()

        amplitude_g = np.max(y)/2
        center_g = x[np.argmax(y)]
        c_g = np.median(y)

        # create parameters with initial values
        params = model.make_params(amplitude=amplitude_g, center=center_g,
                                   sigma=abs(e_axis_step)*2,
                                   gamma=abs(e_axis_step)*2, c=c_g)

        # maybe place bounds on some parameters
        params['center'].min = np.min(x)
        params['center'].max = np.max(x)
        params['sigma'].min = abs(e_axis_step)
        params['sigma'].max = abs(e_axis_step)*200
        params['gamma'].min = abs(e_axis_step)
        params['gamma'].max = abs(e_axis_step)*200
        if amplitude_g != 0:
            params['amplitude'].min = amplitude_g/10
            params['amplitude'].max = amplitude_g*100
        if np.min(y) + np.max(y) != 0:
            params['c'].min = np.min(y)
            params['c'].max = np.max(y)

        # do the fit, print out report with results
        result = model.fit(y, params, x=x)
        print(result.fit_report())

        self.center = np.around(result.params['center'].value, 2)
        sigma = result.params['sigma'].value
        gamma = result.params['gamma'].value
        amplitude = result.params['amplitude'].value
        c = result.params['amplitude'].value
        self.fwhm = np.around(result.params['fwhm'].value, 2)

        center_std = result.params['center'].stderr
        sigma_std = result.params['sigma'].stderr
        gamma_std = result.params['gamma'].stderr
        amplitude_std = result.params['amplitude'].stderr
        c_std = result.params['amplitude'].stderr
        fwhm_std = result.params['fwhm'].stderr

        self.x_fit = np.arange(x[0], x[-1], np.diff(x)[0]/10)
        self.y_fit = result.eval(x=self.x_fit)
        self.fit = True

    def dif_plot(self):
        try:
            with open('config.json', 'r') as json_file:
                config = json.load(json_file)
        except FileNotFoundError:
            with open('packages/config.json', 'r') as json_file:
                config = json.load(json_file)
        config = json.dumps(config)
        config = json.loads(config,
                            object_hook=lambda d: SimpleNamespace(**d))
        self.plot_dif = True
        magn = config.t_dif_magn
        dif_labels = []
        t_cut_plot = self.cuts
        counter = 0
        t_cut_plot_dif = []
        for i in range(len(t_cut_plot)):
            if counter == 0:
                reference = t_cut_plot[i]
            else:
                dif_line = [a - b for a, b in zip(t_cut_plot[i], reference)]
                if magn > 1 or magn < 1:
                    dif_line = [a*magn for a in dif_line]
                    dif_labels.append(f'Difference {self.var_n}$_{counter+1}$-{self.var_n}$_1$ x {magn}')
                else:
                    dif_labels.append(f'Difference {self.var_n}$_{counter+1}$-{self.var_n}$_1$')
                t_cut_plot_dif.append(np.array(dif_line))
            counter += 1
        self.dif_cuts = t_cut_plot_dif
        self.dif_labels = dif_labels

    def waterfall(self):
        try:
            with open('config.json', 'r') as json_file:
                config = json.load(json_file)
        except FileNotFoundError:
            with open('packages/config.json', 'r') as json_file:
                config = json.load(json_file)
        config = json.dumps(config)
        config = json.loads(config,
                            object_hook=lambda d: SimpleNamespace(**d))
        cut_y_max = np.nanmax(self.cuts)
        cut_y_min = np.nanmin(self.cuts)
        offset = (cut_y_max - cut_y_min)*config.t_wat_offset
        t_cut_plot = self.cuts
        for i in range(len(t_cut_plot)-1):
            t_cut_plot_wf_1 = np.delete(np.array(t_cut_plot), -1, axis=0)
            t_cut_plot_wf_2 = np.delete(np.array(t_cut_plot), 0, axis=0)
            t_cut_plot_wf_delta = t_cut_plot_wf_2 - t_cut_plot_wf_1
            t_cut_plot_wf_delta = np.abs(np.min(t_cut_plot_wf_delta, axis=1))
            t_cut_plot_wf_delta = list(t_cut_plot_wf_delta)
            t_cut_plot_wf = []
            counter = 0
            for i in t_cut_plot:
                if counter == 0:
                    t_cut_plot_wf.append(i)
                else:
                    line = [a + t_cut_plot_wf_delta[counter - 1] for a in i]
                    t_cut_plot_wf.append(line)
                counter += 1
            t_cut_plot = t_cut_plot_wf

        t_cut_plot_wf = []
        if offset > 0:
            counter = 0
            for i in t_cut_plot:
                if counter == 0:
                    t_cut_plot_wf.append(i)
                else:
                    line = [a + offset*counter for a in i]
                    t_cut_plot_wf.append(line)
                counter += 1
            t_cut_plot = t_cut_plot_wf
        self.cuts = list(t_cut_plot)

    def savgol_smooth(self, window_length=3, polyorder=1, cycles=1):
        '''
        Method which applies Savitzky–Golay filter to all the stored slices.
        '''
        cuts = []
        for i in self.cuts:
            cut = i
            for j in range(cycles):
                cut = savgol_filter(cut, window_length, polyorder,
                                    mode='nearest')
            cuts.append(cut)
        self.cuts = cuts

    def derivative(self, cycles=3):
        '''
        Method which converts curves of slices to their derivatives.
        It can help to find time zero for slices with exponential behavior.
        '''
        cuts = []
        for i in self.cuts:
            cut = i
            cut = np.abs(np.gradient(cut))
            cuts.append(cut)
        self.cuts = cuts
        self.arb_u = True

    def norm_01(self):
        '''
        Method for normalization of slices to zero to one intensity.
        '''
        cuts = []
        for i in self.cuts:
            cut = i
            norm = np.min(cut)
            cut = cut - norm
            norm = np.max(cut)
            cut = cut/norm
            cuts.append(cut)
        self.cuts = cuts
        self.arb_u = True

    def norm_11(self):
        '''
        Method for normalization of delay-energy map to minus one to one
        intensity range. Either high or low limit absolute value is one.
        The other limit is scaled accordingly.
        It suits well for the difference plot.
        '''
        pos_norm = np.max(self.cuts, axis=0)
        pos_norm = np.max(pos_norm)
        neg_norm = np.min(self.cuts, axis=0)
        neg_norm = np.min(neg_norm)
        norm = [neg_norm, pos_norm]
        norm = np.max(np.abs(norm))
        cuts = []
        for i in self.cuts:
            cut = i
            cut = cut/norm
            cuts.append(cut)
        self.cuts = cuts
        self.arb_u = True

    def axs_plot(self, axs):
        '''
        Method for creating matplotlib axes for map_cut slices.
        '''
        # Loading configs from json file.
        try:
            with open('config.json', 'r') as json_file:
                config = json.load(json_file)
        except FileNotFoundError:
            with open('packages/config.json', 'r') as json_file:
                config = json.load(json_file)
        config = json.dumps(config)
        config = json.loads(config,
                            object_hook=lambda d: SimpleNamespace(**d))
        if self.plot_dif is True:
            self.cut_y_max = np.nanmax(self.cuts+self.dif_cuts)
            self.cut_y_min = np.nanmin(self.cuts+self.dif_cuts)
        else:
            self.cut_y_max = np.nanmax(self.cuts)
            self.cut_y_min = np.nanmin(self.cuts)
        self.cut_y_tick = (self.cut_y_max - self.cut_y_min)/config.t_n_ticks_y
        if self.cut_y_tick == 0:
            self.cut_y_tick = 1
        if self.cut_y_tick < 1:
            self.cut_y_tick_decimal = 1
        else:
            self.cut_y_tick_decimal = 0
        self.cut_y_tick = round(self.cut_y_tick, self.cut_y_tick_decimal)

        self.cut_x_max = np.nanmax(self.coords)
        self.cut_x_min = np.nanmin(self.coords)
        self.cut_x_tick = (self.cut_x_max - self.cut_x_min)/config.t_n_ticks_x
        self.cut_x_tick_decimal = 0
        self.cut_x_tick = round(self.cut_x_tick, self.cut_x_tick_decimal)
        if self.cut_x_tick == 0:
            self.cut_x_tick = 1

        axs.set_title(f'Cuts across {self.axis}', pad=15,
                      fontsize=config.font_size_axis*1.2,
                      fontweight="light")

        if self.axis == 'Energy axis':
            axs.set_xlabel('Delay (ps)', labelpad=10,
                           fontsize=config.font_size_axis)
            var_n = 'E'
        else:
            axs.set_xlabel(f'{self.e_axis} (eV)', labelpad=10,
                           fontsize=config.font_size_axis)
            var_n = 'T'

        if self.fit is False:
            for i, cut in enumerate(self.cuts):
                label = f'{var_n}$_{i+1}$ = {self.positions[i]} {self.units}, '
                label = label + f'd{var_n}$_{i+1}$ = {self.deltas[i]} {self.units}'
                axs.plot(self.coords, cut, config.line_type_d,
                         color=color_dict[i],
                         label=label,
                         markersize=config.marker_size_d,
                         linewidth=config.line_width_d,
                         alpha=config.line_op_d/100)
        else:
            label = f'{var_n} = {self.positions[0]} {self.units}, '
            label = label + f'd{var_n} = {self.deltas[0]} {self.units}'
            axs.plot(self.coords, self.cuts[0], 'o',
                     markerfacecolor='none',
                     markeredgewidth=config.line_width_d*2,
                     color=color_dict[0],
                     label=label,
                     markersize=config.marker_size_d*2,
                     alpha=config.line_op_d/100)

            label = f'Fit: {self.var_n_r} = {self.center} {self.units_r}, '
            label = label + f'FWHM = {self.fwhm} {self.units_r}'
            axs.plot(self.x_fit, self.y_fit, '-',
                     color=color_dict[1],
                     label=label,
                     linewidth=config.line_width_d*4,
                     alpha=config.line_op_d/100)
            self.fit = False

        if self.plot_dif is True:
            for i, cut in enumerate(self.dif_cuts):
                label = self.dif_labels[i]
                axs.plot(self.coords, cut, config.line_type_d,
                         color=color_dict[i],
                         label=label,
                         markersize=config.marker_size_d,
                         linewidth=config.line_width_d,
                         alpha=config.line_op_d/100)

        if self.arb_u is True:
            axs.set_ylabel('Intensity (arb. units)', labelpad=10,
                           fontsize=config.font_size_axis)
        else:
            axs.set_ylabel('Intensity (counts)', labelpad=10,
                           fontsize=config.font_size_axis)
        # y axis
        axs.yaxis.set_major_locator(MultipleLocator(self.cut_y_tick))
        axs.set_ylim(self.cut_y_min-self.cut_y_tick/2,
                     self.cut_y_max + self.cut_y_tick)
        axs.yaxis.set_minor_locator(MultipleLocator(self.cut_y_tick /
                                                    config.t_n_ticks_minor))
        # x axis
        axs.xaxis.set_major_locator(MultipleLocator(self.cut_x_tick))
        axs.xaxis.set_minor_locator(MultipleLocator(self.cut_x_tick /
                                                    config.t_n_ticks_minor))
        axs.set_xlim(self.cut_x_min, self.cut_x_max)

        axs.grid(which='major', axis='both', color='lightgrey',
                 linestyle=config.line_type_grid_d,
                 alpha=config.line_op_grid_d/100,
                 linewidth=config.line_width_grid_d)
        axs.tick_params(axis='both', which='major',
                        length=config.t_tick_length,
                        width=config.t_tick_length/4)
        axs.tick_params(axis='both', which='minor',
                        length=config.t_tick_length/1.5,
                        width=config.t_tick_length/4)
        if self.t_axis == 'Delay stage values' and self.axis == 'Energy axis':
            axs.invert_xaxis()
        if self.e_axis == 'Binding energy' and self.axis == 'Time axis':
            axs.invert_xaxis()

    def save_cut_dat(self):
        '''
        Method for saving the delay-energy map cuts from visualization
        to ASCII format.
        One can find the saved result in the 'ASCII_output' folder.
        '''
        arr = np.array(self.cuts)
        length = arr.shape[0]
        ts = calendar.timegm(gmtime())
        date_time = datetime.fromtimestamp(ts)
        str_date_time = date_time.strftime("%d.%m.%Y_%H-%M-%S")
        path = self.file_dir + os.sep + 'ASCII_output'
        if os.path.isdir(path) is False:
            os.mkdir(path)
        path = path + os.sep + 'Cuts'
        if os.path.isdir(path) is False:
            os.mkdir(path)
        path = path + os.sep + str_date_time + os.sep
        if os.path.isdir(path) is False:
            os.mkdir(path)
        with open(path+"Summary.txt", "w") as text_file:
            text_file.write(f'Loaded runs: {self.run_num_o}\n')
            text_file.write(f'Cuts across: {self.axis}\n')
            positions_str = [str(i) for i in self.positions]
            positions_str = ', '.join(positions_str)
            deltas_str = [str(i) for i in self.deltas]
            deltas_str = ', '.join(deltas_str)
            text_file.write(f'Cut positions: {positions_str} {self.units}\n')
            text_file.write(f'Cut widths: {deltas_str} {self.units}\n')
            text_file.write('Delay-energy map parameters:\n')
            text_file.write(f'Energy step: {self.energy_step} eV\n')
            if self.ordinate == 'delay':
                text_file.write(f'Delay step: {self.delay_step} ps\n')
            elif self.ordinate == 'MB_ID':
                text_file.write(f'MicroBunch step: {self.delay_step} u.\n')
            text_file.write(f'Energy axis: {self.e_axis}\n')
            if self.ordinate == 'delay':
                text_file.write(f'Time axis: {self.t_axis}\n')
            elif self.ordinate == 'MB_ID':
                text_file.write('Time axis: MicroBunch ID\n')

        for i in range(length):
            x = self.coords
            x = list(x)
            if self.axis == 'Energy axis':
                x = [read_file.rounding(i, self.delay_step) for i in x]
            else:
                x = [read_file.rounding(i, self.energy_step) for i in x]
            x = np.array(x)
            x = np.expand_dims(x, axis=0)
            y = arr[i]
            y = np.expand_dims(y, axis=0)
            out = np.append(x, y, axis=0)
            out = np.rot90(out)

            file_full = path
            cut_pos = np.around(self.positions[i], 2)
            cut_delta = np.around(self.deltas[i], 2)
            if self.t_axis == 'Delay stage values':
                order = length - 1 - i
            else:
                order = i
            if len(str(order)) == len(str(length)):
                order = str(order)
            else:
                order = str(order)
                for j in range(len(str(length))-len(str(order))):
                    order = '0' + order
            file_full = file_full + f'{order}_{cut_pos} (d{cut_delta})'
            if self.ordinate == 'delay':
                file_full = file_full + ' ps.dat'
            elif self.ordinate == 'MB_ID':
                file_full = file_full + ' u.dat'
            np.savetxt(file_full, out, delimiter='    ')
            print(f"Saved as {file_full}")


class plot_files:
    '''
    The class for creating matplotlib plots from a list of objects.
    Uses axs_plot methods of the objects.
    '''

    def __init__(self, objects, direction='down', dpi=300,
                 fig_width=7, fig_height=5):
        # Loading configs from json file.
        try:
            with open('config.json', 'r') as json_file:
                config = json.load(json_file)
        except FileNotFoundError:
            with open('packages/config.json', 'r') as json_file:
                config = json.load(json_file)
        config = json.dumps(config)
        config = json.loads(config,
                            object_hook=lambda d: SimpleNamespace(**d))
        self.direction = direction

        if not isinstance(objects, list):
            objects = [objects]

        try:
            fig_number = len(objects)
        except TypeError:
            fig_number = 1

        matplotlib.rcParams.update({'font.size': config.font_size,
                                    'font.family': config.font_family,
                                    'axes.linewidth': config.axes_linewidth})

        fig, axs = plt.subplots(nrows=fig_number, ncols=1, sharex=False,
                                figsize=(fig_width,
                                         fig_height*fig_number),
                                dpi=dpi,
                                gridspec_kw={'hspace': 0.5*5/fig_height}
                                )

        for fig_p, object_i in enumerate(objects):
            fig_p_real = fig_p
            if direction == 'up':
                fig_p += 1
                fig_p = -fig_p

            if fig_number == 1:
                object_i.axs_plot(axs)
            else:
                object_i.axs_plot(axs[fig_p])

        self.axs = axs
        self.fig = fig

    def span_plot(self, cut_obj):
        self.delay_energy_map_plot = cut_obj.delay_energy_map_plot
        try:
            self.varied_y_step = cut_obj.varied_y_step
        except AttributeError:
            self.varied_y_step = 'ND'
        '''
        Method which highlights regions on delay-energy map related to
        slices of map_cut objects.
        '''
        for i in self.fig.axes:
            if i.yaxis.get_label()._text.split(' ')[0] != 'Intensity':
                for counter, position in enumerate(cut_obj.positions):
                    if cut_obj.map_show[counter]:
                        limit_1 = position - cut_obj.deltas[counter]/2
                        limit_2 = position + cut_obj.deltas[counter]/2
                        if cut_obj.axis == 'Energy axis':
                            i.axvspan(limit_1, limit_2,
                                      facecolor=color_dict[counter],
                                      alpha=0.25)
                            for j in [position, limit_1, limit_2]:
                                i.axvline(x=j, color=color_dict[counter],
                                          linewidth=2, zorder=10, alpha=0.4,
                                          linestyle='--')
                        else:
                            if self.varied_y_step != 'ND':
                                if self.varied_y_step is True:
                                    coord = self.delay_energy_map_plot.coords['Delay']
                                    position = coord.sel(Delay=position, method="nearest")
                                    limit_1 = coord.sel(Delay=limit_1, method="nearest")
                                    limit_2 = coord.sel(Delay=limit_2, method="nearest")
                                    position = coord.where(coord==position,drop=True)
                                    limit_1 = coord.where(coord==limit_1,drop=True)
                                    limit_2 = coord.where(coord==limit_2,drop=True)
                                    position = position['Delay index'].values
                                    limit_1 = limit_1['Delay index'].values
                                    limit_2 = limit_2['Delay index'].values
                            i.axhspan(limit_1, limit_2,
                                      facecolor=color_dict[counter],
                                      alpha=0.25)
                            for j in [position, limit_1, limit_2]:
                                i.axhline(y=j, color=color_dict[counter],
                                          linewidth=2, zorder=10, alpha=0.4,
                                          linestyle='--')

    def legend_plot(self):
        '''
        Method for adding a legend to the figure.
        '''
        for i in self.fig.axes:
            if i.yaxis.get_label()._text.split(' ')[0] == 'Intensity':
                if i.xaxis.get_label()._text != '':
                    i.legend(loc='best',
                             fontsize=config.font_size-2, markerscale=2)


if __name__ == "__main__":
    # Loading configs from json file.
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)
    try:
        with open('config.json', 'r') as json_file:
            config = json.load(json_file)
    except FileNotFoundError:
        with open('packages/config.json', 'r') as json_file:
            config = json.load(json_file)
    config = json.dumps(config)
    config = json.loads(config,
                        object_hook=lambda d: SimpleNamespace(**d))

    file_dir = 'D:/Data/Extension_2021_final'
    run_numbers = [37378, 37379, 37380, 37381, 37382, 37383]
    # run_numbers = [37333, 37334, 37335, 37336, 37337, 37341, 37342, 37343, 37344, 37346, 37347, 37348, 37327,37328,37329,37330]
    # run_numbers = [37292,37293,37294,37299,37300,37302,37303,37312,37313,37314,37318,37319,37320]
    run_numbers = [36732,36733,36734,36778,36779,36780,36781,36782,36783]
    # run_numbers = [36732,36733,36734]
    run_numbers = [37262,37266]
    b = create_batch(file_dir, run_numbers, DLD='DLD4Q')
    for i in b.batch_list:
        i.create_map(energy_step=0.01, delay_step=0.1, ordinate='delay',
                     save='off')
        # i.set_BE()
    b.create_map()
    b.norm_total_e()
    # b.set_BE()
    # c = map_cut(b, [101], [0.3], axis='Energy axis')
    plot_files([b])
else:
    # Loading configs from json file.
    try:
        with open('config.json', 'r') as json_file:
            config = json.load(json_file)
    except FileNotFoundError:
        with open('packages/config.json', 'r') as json_file:
            config = json.load(json_file)
    config = json.dumps(config)
    config = json.loads(config,
                        object_hook=lambda d: SimpleNamespace(**d))
