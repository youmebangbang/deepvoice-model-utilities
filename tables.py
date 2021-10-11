import numpy as np
import dearpygui.dearpygui as dpg


class Tables:
    def __init__(self):
        
        self.table_array = np.empty([0,2])
        with dpg.table(row_background=True, borders_innerH=True, borders_outerH=True, borders_innerV=True,
                            borders_outerV=True, parent="inference_tab", header_row=True, width=1100, height=400, tag="infer_table"):
            dpg.add_table_column(width_fixed=True, init_width_or_weight=700, parent="infer_table", label='TEXT')
            dpg.add_table_column(width_fixed=True, init_width_or_weight=200, parent="infer_table", label='AUDIO FILE')
            dpg.add_table_column(width_fixed=True, init_width_or_weight=200, parent="infer_table", label='OPTIONS')

    def show_table(self):
        # clear table
        dpg.delete_item("infer_table")
        with dpg.table(row_background=True, borders_innerH=True, borders_outerH=True, borders_innerV=True,
                            borders_outerV=True, parent="inference_tab", header_row=True, width=1100, height=400, tag="infer_table"):
            dpg.add_table_column(width_fixed=True, init_width_or_weight=700, parent="infer_table", label='TEXT')
            dpg.add_table_column(width_fixed=True, init_width_or_weight=200, parent="infer_table", label='AUDIO FILE')
            dpg.add_table_column(width_fixed=True, init_width_or_weight=200, parent="infer_table", label='OPTIONS')        
        print(self.table_array)
        l = len(self.table_array)
        for i in range(0, l):    
            with dpg.table_row(parent="infer_table"):             
                dpg.add_input_text(default_value=str(self.table_array[i][0]), width=700)
                dpg.add_text(str(self.table_array[i][1]))
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Play")
                    dpg.add_button(label="Redo")
                    dpg.add_button(label="Remove")
    def add_entry(self, entry):
        self.table_array = np.vstack((self.table_array, entry))
        self.show_table()