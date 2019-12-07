'''
===========
SimpleLayer
===========

A plugin to allow a quick and simple creation of layers from materials created
from a crystallographic data file (.cif) available at e.g. ICSD
or the chemical formula and mass-density or crystal structure.
Materials are stored for later use.

To use the materials select any layer in the Sample tab and one material in the
Materials tab and than click the blue arrow button. If you select a Stack instead
a new layer will be created with the given material. The plugin will insert
the x-ray and neutron values ( *f* , *b* ) as well as the density and rename the layer
with the given formula.

Written by Artur Glavic
Last Changes 10/11/16
'''

import os, sys
import re
import json
import wx
from wx.lib.mixins.listctrl import ListCtrlAutoWidthMixin
from math import cos, pi, sqrt
from models.utils import UserVars, fp, fw, bc, bw, __bc_dict__ #@UnusedImport
import images as img
from plugins import add_on_framework as framework
from genx.gui_logging import iprint
from .help_modules.materials_db import mdb, Formula, MASS_DENSITY_CONVERSION

mg=None

class Plugin(framework.Template):
    _refplugin=None
    
    @property
    def refplugin(self):
        # check if reflectivity plugin is None or destoryed, try to connect
        if not self._refplugin:
            self._init_refplugin()
        return self._refplugin
    
    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        self.parent = parent
        # on the right side, add a list of materials with their density to selct from
        materials_panel = self.NewDataFolder('Materials')
        materials_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.materials_panel = wx.Panel(materials_panel)
        self.create_materials_list()
        materials_sizer.Add(self.materials_panel, 1, wx.EXPAND | wx.GROW | wx.ALL)
        materials_panel.SetSizer(materials_sizer)
        materials_panel.Layout()

    def _init_refplugin(self):
        try:
            # connect to the reflectivity plugin for layer creation
            self._refplugin = self.parent.plugin_control.plugin_handler.loaded_plugins['Reflectivity']
        except KeyError:
            dlg=wx.MessageDialog(self.materials_panel, 'Reflectivity plugin must be loaded',
                             caption='Information',
                             style=wx.OK|wx.ICON_WARNING)
            dlg.ShowModal()
            dlg.Destroy()
            self._refplugin=None

    def create_materials_list(self):
        '''
          Create a list of materials and it's graphical representation as
          well as a toolbar.
        '''
        self.known_materials=mdb
        # self.tool_panel=wx.Panel(self.materials_panel)
        self.materials_list=MaterialsList(self.materials_panel, self.known_materials)
        self.sizer_vert=wx.BoxSizer(wx.VERTICAL)
        self.materials_panel.SetSizer(self.sizer_vert)

        self.create_toolbar()

        # self.sizer_vert.Add(self.tool_panel, proportion=0, flag=wx.EXPAND, border=5)
        self.sizer_vert.Add((-1, 2))
        self.sizer_vert.Add(self.materials_list, proportion=1, flag=wx.EXPAND , border=5)
        # self.tool_panel.SetSizer(self.sizer_hor)

    def create_toolbar(self):
        self.toolbar = wx.ToolBar(self.materials_panel, style=wx.TB_FLAT|wx.TB_HORIZONTAL)

        dpi_scale_factor=wx.GetDisplayPPI()[0]/96.
        tb_bmp_size=int(dpi_scale_factor*20)

        newid=wx.NewId()
        self.toolbar.AddTool(newid, label='Add',
             bitmap=wx.Bitmap(img.add.GetImage().Scale(tb_bmp_size,tb_bmp_size)),
             shortHelp='Add a material to the list')
        self.materials_panel.Bind(wx.EVT_TOOL, self.material_add, id=newid)

        newid=wx.NewId()
        self.toolbar.AddTool(newid, label='Delete',
             bitmap=wx.Bitmap(img.delete.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
             shortHelp='Delete selected materials')
        self.materials_panel.Bind(wx.EVT_TOOL, self.material_delete, id=newid)

        newid=wx.NewId()
        self.toolbar.AddTool(newid, label='Apply',
             bitmap=wx.Bitmap(img.start_fit.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
             shortHelp='New Layer/Apply to Layer')
        self.materials_panel.Bind(wx.EVT_TOOL, self.material_apply, id=newid)
        
        # self.sizer_hor=wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_vert.Add(self.toolbar, proportion=0, flag=wx.EXPAND, border=2)
        self.toolbar.Realize()


    def material_add(self, event):
        dialog=MaterialDialog(self.parent)
        if dialog.ShowModal()==wx.ID_OK:
            self.materials_list.AddItem(dialog.GetResult())
        dialog.Destroy()

    def material_delete(self, event):
        self.materials_list.DeleteItem()

    def material_apply(self, event):
        index=self.materials_list.GetFirstSelected()
        formula, density=self.known_materials[index]
        try:
            layer=self.get_selected_layer()
        except:
            dlg=wx.MessageDialog(self.materials_panel,
                'You have to select a layer or stack before applying material',
                             caption='Information',
                             style=wx.OK|wx.ICON_INFORMATION)
            dlg.ShowModal()
            dlg.Destroy()
            return
        if layer:
            layer.f=formula.f()
            layer.b=formula.b()
            layer.dens=density
        name=''
        for element, count in formula:
            # get rid of isotope unusable characters
            element=element.replace('{', '').replace('}', '').replace('^', 'i')
            if count==1:
                name+="%s"%(element)
            elif float(count)==int(count):
                name+="%s%i"%(element, count)
            else:
                name+=("%s%s"%(element, count)).replace('.', '_')
        self.set_layer_name(name)
        try:
          self.refplugin.sample_widget.UpdateListbox()
        except AttributeError:
          self.refplugin.sample_widget.Update()

    def get_selected_layer(self):
        layer_idx=self.refplugin.sample_widget.listbox.GetSelection()
        active_layer=self.refplugin.sampleh.getItem(layer_idx)
        if active_layer.__class__.__name__=="Stack":
                # create a new layer to return
            self.refplugin.sampleh.insertItem(layer_idx, 'Layer', 'WillChange')
            active_layer=self.refplugin.sampleh.getItem(layer_idx+1)
        return active_layer

    def set_layer_name(self, name):
        layer_idx = self.refplugin.sample_widget.listbox.GetSelection()
        if self.refplugin.sampleh.names[layer_idx] in ['Amb', 'Sub']:
            return
        active_layer = self.refplugin.sampleh.getItem(layer_idx)
        if active_layer.__class__.__name__ == "Stack":
            # create a new layer to return
            layer_idx += 1
        tmpname = name
        i = 1
        while tmpname in self.refplugin.sampleh.names:
            tmpname = '%s_%i'%(name, i)
            i += 1
        self.refplugin.sampleh.names[layer_idx] = tmpname


class MaterialsList(wx.ListCtrl, ListCtrlAutoWidthMixin):
    '''
    The ListCtrl for the materials data.
    '''
    def __init__(self, parent, materials_list):
        wx.ListCtrl.__init__(self, parent, -1,
             style=wx.LC_REPORT|wx.LC_VIRTUAL)
        ListCtrlAutoWidthMixin.__init__(self)
        self.materials_list = materials_list
        self.parent = parent

        font = self.GetFont()
        font.SetPointSize(9)
        self.SetFont(font)

        # Set list length
        self.SetItemCount(len(materials_list))

        # Set the column headers
        for col, (text, width) in enumerate([
                                             ("Chemical Formula", 80),
                                             ("SLD-n\n[10⁻⁶Å⁻²]", 60),
                                             ("SLD-kα\n[rₑ/Å⁻³]", 60),
                                             ("Density\n[FU/Å³]", 60),
                                             ("Density\n[g/cm³]", 60)
                                             ]):
            self.InsertColumn(col, text, width=width)
        
        self.setResizeColumn(0)

    def OnSelectionChanged(self, evt):
        if not self.toggleshow:
            indices = self._GetSelectedItems()
            indices.sort()
            if not indices == self.show_indices:
                self.data_cont.show_data(indices)
                self._UpdateData('Show data set flag toggled',
                                 data_changed=True)
                # Forces update of list control
                self.SetItemCount(self.data_cont.get_count())
        evt.Skip()

    def OnGetItemText(self, item, col):
        if col == 4:
            return "%.3f" % self.materials_list.dens_mass(item)
        elif col == 3:
            return "%.4f" % self.materials_list.dens_FU(item).real
        elif col == 2:
            return "%.3f" % self.materials_list.SLDx(item).real
        elif col == 1:
            return "%.3f" % self.materials_list.SLDn(item).real
        else:
            formula = self.materials_list[item][0]
            return str(formula)

    def _GetSelectedItems(self):
        ''' _GetSelectedItems(self) --> indices [list of integers]
        Function that yields a list of the currently selected items
        position in the list. In order of selction, i.e. no order.
        '''
        indices = [self.GetFirstSelected()]
        while indices[-1] != -1:
            indices.append(self.GetNextSelected(indices[-1]))

        # Remove the last will be -1
        indices.pop(-1)
        return indices

    def _CheckSelected(self, indices):
        '''_CheckSelected(self, indices) --> bool
        Checks so at least data sets are selcted, otherwise show a dialog box
        and return False
        '''
        # Check so that one dataset is selected
        if len(indices) == 0:
            dlg = wx.MessageDialog(self, 'At least one data set has to be selected', caption='Information',
                                   style=wx.OK | wx.ICON_INFORMATION)
            dlg.ShowModal()
            dlg.Destroy()
            return False
        return True

    def DeleteItem(self):
        index = self.GetFirstSelected()
        item = self.materials_list[index]
        item_formula = ''
        for element, count in item[0]:
            if count == 1:
                item_formula += "%s" % (element)
            elif float(count) == int(count):
                item_formula += "%s%i" % (element, count)
            else:
                item_formula += "%s(%f)" % (element, count)


        # Create the dialog box
        dlg = wx.MessageDialog(self, 'Remove material %s?' % (item_formula),
        caption = 'Remove?', style=wx.YES_NO | wx.ICON_QUESTION)

        # Show the dialog box
        if dlg.ShowModal() == wx.ID_YES:
            self.materials_list.pop(index)
            # Update the list
            self.SetItemCount(len(self.materials_list))

        dlg.Destroy()


    def AddItem(self, item):
        i = 0
        while i < len(self.materials_list) and self.materials_list[i][0] < item[0]:
            i+=1
        self.materials_list.insert(i, item)
        self.SetItemCount(len(self.materials_list))

class MaterialDialog(wx.Dialog):
    """
      Dialog to get material information from chemical formula and atomic density.
      Atomic density can ither be entered manually, by using unit cell parameter,
      by massdensity or by loading a .cif crystallographical file.
    """

    def __init__(self, parent):
        wx.Dialog.__init__(self, parent, name='New Material')
        self.extracted_elements=Formula([])
        self._create_entries()

    def _create_entries(self):
        base_layout = wx.BoxSizer(wx.VERTICAL)

        table = wx.GridBagSizer(5, 2)

        self.formula_entry = wx.TextCtrl(self, size=(100, 25))
        self.formula_entry.Bind(wx.EVT_TEXT, self.OnFormulaChanged)
        table.Add(wx.StaticText(self, label="Formula:"), (0, 0), flag=wx.ALIGN_CENTER)
        table.Add(self.formula_entry, (0, 1), span=(1, 2), flag=wx.EXPAND)

        table.Add(wx.StaticText(self, label="Extracted Elements:"), (1, 0),
                  span=(1, 3), flag=wx.ALIGN_CENTER)
        self.formula_display=wx.TextCtrl(self, size=(150, 100),
                                         style=wx.TE_MULTILINE|wx.TE_READONLY)
#        self.formula_display.Enable(False)
        table.Add(self.formula_display, (2, 0), span=(1, 3), flag=wx.EXPAND)

        table.Add(wx.StaticText(self, label="1. Unit Cell Parameters:"), (3, 0),
                  span=(1, 3), flag=wx.ALIGN_CENTER)
        table.Add(wx.StaticText(self, label="a [Å]"), (4, 0), flag=wx.ALIGN_CENTER)
        self.a_entry=wx.TextCtrl(self, size=(50, 25))
        table.Add(self.a_entry, (5, 0), flag=wx.EXPAND)
        table.Add(wx.StaticText(self, label="b [Å]"), (4, 1), flag=wx.ALIGN_CENTER)
        self.b_entry=wx.TextCtrl(self, size=(50, 25))
        table.Add(self.b_entry, (5, 1), flag=wx.EXPAND)
        table.Add(wx.StaticText(self, label="c [Å]"), (4, 2), flag=wx.ALIGN_CENTER)
        self.c_entry=wx.TextCtrl(self, size=(50, 25))
        table.Add(self.c_entry, (5, 2), flag=wx.EXPAND)

        table.Add(wx.StaticText(self, label="α"), (6, 0), flag=wx.ALIGN_CENTER)
        self.alpha_entry=wx.TextCtrl(self, size=(50, 25))
        self.alpha_entry.SetValue('90')
        table.Add(self.alpha_entry, (7, 0), flag=wx.EXPAND)
        table.Add(wx.StaticText(self, label="β"), (6, 1), flag=wx.ALIGN_CENTER)
        self.beta_entry=wx.TextCtrl(self, size=(50, 25))
        self.beta_entry.SetValue('90')
        table.Add(self.beta_entry, (7, 1), flag=wx.EXPAND)
        table.Add(wx.StaticText(self, label="γ"), (6, 2), flag=wx.ALIGN_CENTER)
        self.gamma_entry=wx.TextCtrl(self, size=(50, 25))
        self.gamma_entry.SetValue('90')
        table.Add(self.gamma_entry, (7, 2), flag=wx.EXPAND)

        table.Add(wx.StaticText(self, label="FUs"), (8, 0), flag=wx.ALIGN_CENTER)
        self.FUs_entry=wx.TextCtrl(self, size=(50, 25))
        self.FUs_entry.SetValue('1')
        table.Add(self.FUs_entry, (9, 0), flag=wx.ALIGN_CENTER)
        cif_button=wx.Button(self, label="From .cif File...")
        cif_button.Bind(wx.EVT_BUTTON, self.OnLoadCif)
        table.Add(cif_button, (8, 1), span=(2, 2), flag=wx.ALIGN_CENTER)

        global mg
        if mg is None:
          try:
            global MPRester
            from pymatgen.ext.matproj import MPRester
            import pymatgen as mg
          except ImportError:
            pass
        if mg is None:
          mg_txt=wx.StaticText(self, label="Install PyMatGen for Online Query")
          table.Add(mg_txt, (10, 0), span=(1, 3), flag=wx.ALIGN_CENTER)
        else:
          mg_button=wx.Button(self, label="Query Online (PyMatGen)")
          mg_button.Bind(wx.EVT_BUTTON, self.OnQuery)
          table.Add(mg_button, (10, 0), span=(1, 3), flag=wx.ALIGN_CENTER)

        for entry in [self.a_entry, self.b_entry, self.c_entry,
                      self.alpha_entry, self.beta_entry, self.gamma_entry,
                      self.FUs_entry]:
            entry.Bind(wx.EVT_TEXT, self.OnUnitCellChanged)


        table.Add(wx.StaticText(self, label="2. Physical Parameter:"), (11, 0),
                  span=(1, 3), flag=wx.ALIGN_CENTER)
        self.mass_density=wx.TextCtrl(self, size=(70, 25))
        self.mass_density.Bind(wx.EVT_TEXT, self.OnMassDensityChange)
        table.Add(wx.StaticText(self, label="Mass Density [g/cm³]:"), (12, 0))
        table.Add(self.mass_density, (12, 1), span=(1, 1))

        table.Add(wx.StaticText(self, label="Result from 1. or 2.:"), (13, 0), span=(1, 3))
        self.result_density=wx.TextCtrl(self, size=(150, 25))
        table.Add(wx.StaticText(self, label="Density [FU/Å³]:"), (14, 0))
        table.Add(self.result_density, (14, 1), span=(1, 2))

        buttons=self.CreateButtonSizer(wx.OK|wx.CANCEL)

        base_layout.Add(table, 1, wx.ALIGN_CENTER|wx.TOP)
        base_layout.Add(buttons, 0, wx.ALIGN_RIGHT)
        self.SetSizerAndFit(base_layout)

    def OnFormulaChanged(self, event):
        text=self.formula_entry.GetValue()
        for ign_char in [" ", "\t", "_", "-"]:
            text=text.replace(ign_char, "")
        if text=="":
            self.extracted_elements=Formula([])
            self.formula_display.SetValue('')
            return
        formula=Formula.from_str(text)
        self.extracted_elements=formula
        self.formula_display.SetValue(formula.describe())
        self.OnMassDensityChange(None)

    def OnUnitCellChanged(self, event):
        params=[]
        for entry in [self.a_entry, self.b_entry, self.c_entry,
                        self.alpha_entry, self.beta_entry, self.gamma_entry,
                        self.FUs_entry]:
            try:
                params.append(float(entry.GetValue()))
            except ValueError:
                return
        if params[3]==90 and params[4]==90 and params[5]==90:
            if params[0]==params[1] and params[0]==params[2]:
                self.density='%s/(%g**3)'%(params[6], params[0])
            else:
                self.density='%s/(%g*%g*%g)'%(params[6], params[0], params[1], params[2])
        else:
            # calculate general unit cell volume (triclinic formula applicable to all structures)
            alpha=params[3]*pi/180.
            beta=params[4]*pi/180.
            gamma=params[5]*pi/180.
            V=params[0]*params[1]*params[2]
            V*=sqrt(1-cos(alpha)**2-cos(beta)**2-cos(gamma)**2+2*cos(alpha)*cos(beta)*cos(gamma))
            self.density='%s/%g'%(params[6], V)
        self.result_density.SetValue(self.density)

    def OnMassDensityChange(self, event):
        fu_mass=self.extracted_elements.mFU()
        try:
            mass_density=float(self.mass_density.GetValue())
        except ValueError:
            return
        self.density="%g*%g/%g"%(mass_density, MASS_DENSITY_CONVERSION, fu_mass)
        self.result_density.SetValue(self.density)

    def OnLoadCif(self, event):
        fd=wx.FileDialog(self,
                         message="Open a (.cif) File...",
                         wildcard='Crystallographic Information File|*.cif;*.CIF|All Files|*',
                         defaultFile='crystal.cif',
                         style=wx.FD_OPEN|wx.FD_CHANGE_DIR)
        if fd.ShowModal()==wx.ID_OK:
            filename=fd.GetPath()
            self.extract_cif(filename)
        fd.Destroy()

    def OnQuery(self, event):
        key='NdHi2bTnJ9WDS1sU'
        a=MPRester(key)
        formula=''
        for element, number in self.extracted_elements:
            if element.startswith('^'):
                element=element.split('}')[-1]
            formula+='%s%g'%(element, number)
        res=a.get_data(formula)
        if type(res) is not list:
            return
        if len(res)>1:
            # more then one structure available, ask for user input to select appropriate
            items=[]
            for i, ri in enumerate(res):
                cs=ri['spacegroup']['crystal_system']
                sgs=ri['spacegroup']['symbol']
                frm=ri['full_formula']
                v=ri['volume']
                dens=ri['density']
                items.append(
                    '%i: %s (%s) | UC Formula: %s\n     Density: %s g/cm³ | UC Volume: %s'%
                    (i+1, sgs, cs, frm, dens, v))
                if ri['tags'] is not None:
                    items[-1]+='\n     '+';'.join(ri['tags'][:3])
            dia=wx.SingleChoiceDialog(self,
                                      'Several entries have been found, please select appropriate:',
                                      'Select correct database entry',
                                      items)
            if not dia.ShowModal()==wx.ID_OK:
                return None
            res=res[dia.GetSelection()]
        else:
            res=res[0]
        return self.analyze_cif(res['cif'])

    def GetResult(self):
        return (self.extracted_elements, self.result_density.GetValue())

    def extract_cif(self, filename):
        """
          Try to get unit cell and formula unit information from a .cif file.
        """
        if not os.path.exists(filename):
            return
        txt=open(filename).read()
        return self.analyze_cif(txt)

    def analyze_cif(self, txt):
        cell_params=[1., 1., 1., 90., 90., 90., 1.]
        composition=''
        file_lines=txt.splitlines()
        for line in file_lines:
            if line.startswith('_cell_length_a'):
                cell_params[0]=round(float(line.split()[1].split('(')[0]), 3)
            if line.startswith('_cell_length_b'):
                cell_params[1]=round(float(line.split()[1].split('(')[0]), 3)
            if line.startswith('_cell_length_c'):
                cell_params[2]=round(float(line.split()[1].split('(')[0]), 3)
            if line.startswith('_cell_angle_alpha'):
                cell_params[3]=round(float(line.split()[1].split('(')[0]), 3)
            if line.startswith('_cell_angle_beta'):
                cell_params[4]=round(float(line.split()[1].split('(')[0]), 3)
            if line.startswith('_cell_angle_gamma'):
                cell_params[5]=round(float(line.split()[1].split('(')[0]), 3)
            if line.startswith('_cell_formula_units_Z'):
                cell_params[6]=int(float(line.split()[1]))
            if line.startswith('_chemical_formula_structural'):
                composition=line.strip().split(None, 1)[1].replace(")", "").replace("(", "").replace("'", "").replace('"', '')
            if line.startswith('_chemical_formula_sum') and composition=='':
                composition=line.strip().split(None, 1)[1].replace("'", "").replace('"', '')
        self.formula_entry.SetValue(composition)
        self.OnFormulaChanged(None)
        for value, entry in zip(cell_params,
                                [self.a_entry,
                                 self.b_entry,
                                 self.c_entry,
                                 self.alpha_entry,
                                 self.beta_entry,
                                 self.gamma_entry,
                                 self.FUs_entry]):
            entry.SetValue(str(value))
        self.OnUnitCellChanged(None)
