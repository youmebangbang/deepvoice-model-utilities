import dearpygui.dearpygui as dpg
import os
import shutil

#Tacotron imports
import os
from dearpygui.dearpygui import get_value
import sys
import numpy as np
import torch
sys.path.append('tacotron2/waveglow/')
sys.path.insert(1, '/tacotron2')
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser
import subprocess
import threading
import time
from pydub import AudioSegment 
from pydub.playback import play
import soundfile as sf
import simpleaudio as sa

dpg.create_context()


class Trainer():
    def __init__(self):
        self.process = None
        self.is_training_running = False
        self.t1 = None
        self.t2 = None     
        self.hifigan_checkpoint_name = None
        self.hifigan_project_name = None 
        self.tensorboard_process = None
    
    def set_tensorboard_process(self, p):
        self.tensorboard_process = p

    def get_tensorboard_process(self):
        return self.tensorboard_process

    def set_hifigan_checkpoint_name(self, path):
        self.hifigan_checkpoint_name = path    

    def get_hifigan_checkpoint_name(self):
        return self.hifigan_checkpoint_name

    def set_hifigan_project_name(self, path):
        self.hifigan_project_name = path    

    def get_hifigan_project_name(self):
        return self.hifigan_project_name

    def is_running(self):
        return self.is_training_running

    def train_taco(self):
        dpg.set_value("shell_output_tacotron2", None)

    def train_waveglow(self):
        pass

    def train_hifigan(self):
        self.is_training_running = True
        os.chdir("hifi-gan")      

        def start_process():
            print(self.hifigan_checkpoint_name)
            print(self.hifigan_project_name)
           
            self.process =  subprocess.Popen(['python', '-u', 'train.py', '--checkpoint_path', self.hifigan_checkpoint_name,
            '--input_training_file', self.hifigan_project_name + '/' + 'training.csv', '--input_validation_file', self.hifigan_project_name + '/' + 'validation.csv', '--input_wavs_dir', self.hifigan_project_name + '/' + 'wavs'] , stdout=subprocess.PIPE)  
                       
            # self.process =  subprocess.Popen(['python', '-u', 'train.py', '--checkpoint_path', 'attenborough.model',
            # '--input_training_file', 'training.csv', '--input_validation_file', 'validation.csv', '--input_wavs_dir', 'wavs'] , stdout=subprocess.PIPE)  
           
            # self.process =  subprocess.run(['python', '-u', 'train.py', '--checkpoint_path', 'attenborough.model',
            # '--input_training_file', 'training.csv', '--input_validation_file', 'validation.csv', '--input_wavs_dir', 'wavs'] , capture_output=True, text=True)  

            os.chdir("../")

            while self.is_training_running:
                time.sleep(.01)
                out = self.process.stdout.readline().decode('utf-8')
                if out: 
                    print(out)
                    dpg.set_value("shell_output_hifigan", out)

        self.t1 = threading.Thread(target=start_process)

        self.t1.start()

        print("HifiGan training started!")
        dpg.set_value("shell_output_hifigan", "HifiGan training started. Check console for more information.")

    def stop_training_hifigan(self):
        print("\nStopping hifigan training... waiting for epoch to end...")
        with open("training_status.txt", 'w') as f:
            f.write("terminate")
        if self.process:
            self.process.kill()
            time.sleep(1)
            self.is_training_running = False
            dpg.set_value("shell_output_hifigan", "training stopped.")
            print("Training stopped successfully.")
            if os.path.exists("hifi-gan/training_status.txt"):
                f = open("hifi-gan/training_status.txt", 'w')
                f.close()                                   
            # os.chdir("../")    

    def stop_training_tacotron2(self):
        pass
        # with open ("hifi-gan/training_status.txt", 'w') as f:
        #     pass
        # if self.process:
        #     self.process.kill()
        #     print("hifigan training has ended")
        #     dpg.set_value("shell_output_tacotron2", "training stopped.")
       
    def stop_training_waveglow(self):
        pass
        # with open ("waveglow/training_status.txt", 'w') as f:
        #     pass
        # if self.process:
        #     self.process.kill()
        #     print("hifigan training has ended")
        #     dpg.set_value("shell_output_waveglow", "training stopped.")

class Inferer():
    def __init__(self):
        self.taco_model_path = None
        self.hifigan_model_path = None
        self.waveglow_model_path = None
        self.taco_model = None
        self.waveglow_model = None
        self.hifigan_model = None
        self.denoiser = None
        self.text_file_path = None
        self.file_count = 0       
        self.table_array = np.empty([0,2])

        with dpg.table(pos=(0, 0), resizable=False, scrollY=True, row_background=True, borders_innerH=True, borders_outerH=True, borders_innerV=True,
                            borders_outerV=True, parent="infer_table_window", header_row=True, width=1325, height=400, tag="infer_table"):
            dpg.add_table_column(width_fixed=True, init_width_or_weight=875, parent="infer_table", label='TEXT')
            dpg.add_table_column(width_fixed=True, init_width_or_weight=250, parent="infer_table", label='AUDIO FILE')
            dpg.add_table_column(width_fixed=True, init_width_or_weight=200, parent="infer_table", label='OPTIONS')
    
    def show_table(self):
        # clear table
        dpg.delete_item("infer_table")
        with dpg.table(pos=(0, 0), resizable=False, scrollY=True, row_background=True, borders_innerH=True, borders_outerH=True, borders_innerV=True,
                            borders_outerV=True, parent="infer_table_window", header_row=True, width=1325, height=400, tag="infer_table"):
            
            dpg.add_table_column(width_fixed=True, init_width_or_weight=875, label='TEXT')
            dpg.add_table_column(width_fixed=True, init_width_or_weight=250, label='AUDIO FILE')
            dpg.add_table_column(width_fixed=True, init_width_or_weight=200, label='OPTIONS')        
        l = len(self.table_array)
        for i in range(0, l): 
            with dpg.table_row(parent="infer_table"): 
                if dpg.does_alias_exist("input_text_" + str(i)):
                    dpg.remove_alias("input_text_" + str(i))
                    dpg.remove_alias("wav_path_" + str(i))
                dpg.add_input_text(tag="input_text_" + str(i), default_value=str(self.table_array[i][0]), width=850)
                dpg.add_text(str(self.table_array[i][1]), tag="wav_path_" + str(i))

                a_path = str(self.table_array[i][1])
                entry_info = {
                    "rank": i,
                    "text": dpg.get_value("input_text_" + str(i)),
                    "wav_path": a_path
                }

                with dpg.group(horizontal=True):
                    dpg.add_button(label="Play", callback=self.callback_play_entry, user_data = a_path)
                    dpg.add_button(label="Redo", callback=self.callback_redo_entry, user_data = entry_info)
                    dpg.add_button(label="Remove", callback=self.callback_remove_entry, user_data = entry_info)
        

    def add_entry(self, entry):
        self.table_array = np.vstack((self.table_array, entry))
        # print(self.table_array)
        self.show_table()

    def callback_play_entry(self, sender, app_data, user_data):
        self.stop()
        a = AudioSegment.from_file(user_data)    
        t_play = threading.Thread(target=self.play, args=(a,))
        t_play.start()

    def callback_redo_entry(self, sender, app_data, user_data):
        status = dpg.get_value("infer_status_text")
        dpg.set_value("infer_status_text", status + "\nRe-inferring entry with updated text.")
        dpg.set_y_scroll("infer_status_window", 1000)
        self.stop()
        text = dpg.get_value("input_text_" + str(user_data["rank"]))
        #run inference again
        result = self.run_inference(text, "text_input", dpg.get_value("infer_project_name") + "/wavs_out/" + str(user_data["rank"]) + ".wav")
        a = AudioSegment.from_file(result[0])    
        t_play = threading.Thread(target=self.play, args=(a,))
        t_play.start()      

    def callback_remove_entry(self, sender, app_data, user_data):
        status = dpg.get_value("infer_status_text")
        dpg.set_value("infer_status_text", status + "\nRemoving entry")
        dpg.set_y_scroll("infer_status_window", 1000)      
        self.stop()    
        new_array = np.empty([0,2])
        rank = user_data["rank"]
        td = inferer.get_table_array()
        for i, r in enumerate(td):
            if not i == rank:
                new_array = np.vstack((new_array, r))
               
        os.remove(user_data["wav_path"])
        inferer.set_table_array(new_array)
        inferer.show_table()

        
    
    def play(self, data):
        wav = data            
        sa.play_buffer(
            wav.raw_data,
            num_channels=wav.channels,
            bytes_per_sample=wav.sample_width,
            sample_rate=wav.frame_rate
        )

    def stop(self):
        sa.stop_all()

    def set_taco_model_path(self, path):
        self.taco_model_path = path
    def set_hifigan_model_path(self, path):
        self.hifigan_model_path = path
    def set_waveglow_model_path(self, path):
        self.waveglow_model_path = path        
    def set_text_file_path(self, path):
        self.text_file_path = path
    def set_table_array(self, a):
        self.table_array = a

    def get_text_file_path(self):
        return self.text_file_path
    def get_hifigan_model_path(self):
        return self.hifigan_model_path
    def get_waveglow_model_path(self):
        return self.waveglow_model_path    
    def get_taco_model_path(self):
        return self.taco_model_path
    def get_table_array(self):
        return self.table_array

    def run_inference(self, input_text, mode, wav_path):
        os.makedirs(dpg.get_value("infer_project_name") + "/wavs_out", exist_ok=True)

        hparams = create_hparams()
        hparams.sampling_rate = 22050
        #hparams change dropouts!  
        hparams.p_attention_dropout = 0
        hparams.p_decoder_dropout = 0
        hparams.max_decoder_steps = 10000

        chosen_model = dpg.get_value("infer_model_radio")
   
        self.taco_model = load_model(hparams)
        self.taco_model.load_state_dict(torch.load(self.taco_model_path)['state_dict'])
        _ = self.taco_model.cuda().eval().half()

        if mode == "text_input":
            text = input_text
            sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
            sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
            mel_outputs, mel_outputs_postnet, _, alignments = self.taco_model.inference(sequence)
           
            #Was waveglow or hifigan used
            if chosen_model == "Use Hifi-Gan model":
                print("Using Hifi-Gan model")
                # Clear hifigan mel directories
                if os.path.exists("hifi-gan/test_mel_files"):
                    shutil.rmtree("hifi-gan/test_mel_files")
                if os.path.exists("hifi-gan/generated_files_from_mel"):
                    shutil.rmtree("hifi-gan/generated_files_from_mel")
                
                os.makedirs("hifi-gan/test_mel_files")
                mel_out = mel_outputs_postnet.cpu().detach().numpy()
                np.save("hifi-gan/test_mel_files/mel.data", mel_out)

                tp = subprocess.run(['python', '-u', 'hifi-gan/inference_e2e.py', '--checkpoint_file', self.hifigan_model_path, '--output_dir', 'hifi-gan/generated_files_from_mel', '--input_mels_dir', 'hifi-gan/test_mel_files'])
                #copy file
                wav_name =  dpg.get_value("infer_project_name") + "/wavs_out/" + str(self.file_count) + ".wav"
                shutil.move("hifi-gan/generated_files_from_mel/mel.data_generated_e2e.wav", wav_name)
                inferer.file_count += 1
                return [wav_name]

            elif chosen_model == "Use Waveglow model":
                print("Using Waveglow model")
                #waveglow
                self.waveglow_model = torch.load(self.waveglow_model_path)['model']
                self.waveglow_model.cuda().eval().half()
                for k in self.waveglow_model.convinv:
                    k.float()
                self.denoiser = Denoiser(self.waveglow_model) 

                with torch.no_grad():
                    audio = self.waveglow_model.infer(mel_outputs_postnet, sigma=1)
                audio_denoised = self.denoiser(audio, strength=0.02)[:, 0]
                audioout = audio_denoised[0].data.cpu().numpy()
                #audioout = audio[0].data.cpu().numpy()
                audioout32 = np.float32(audioout)  
                if wav_path:
                    wav_name = wav_path
                else:  
                    wav_name = dpg.get_value("infer_project_name") + '/wavs_out/' + str(self.file_count) + '.wav'
                sf.write(wav_name, audioout32, 22050)
                inferer.file_count += 1
                return [wav_name]

        elif mode == "text_input_file":
            #break text apart and infer each phrase.
            #get max word length of phrase if no punctuation
            import re
            input_text = input_text.strip('\n')
            input_text = input_text.strip('\t')
            phrase_splits = re.split(r'(?<=[\.\!\?])\s*', input_text)   #split on white space between sentences             
            phrase_splits = list(filter(None, phrase_splits))  #remove empty splits
            p_count = len(phrase_splits)
            if phrase_splits:
                result = []
                for i, p in enumerate(phrase_splits):
                    
                    status = dpg.get_value("infer_status_text")
                    dpg.set_value("infer_status_text", status + "\nInferring cut {} of {}.".format(i+1, p_count))
                    dpg.set_y_scroll("infer_status_window", 1000)

                    if chosen_model == "Use Hifi-Gan model":
                        print("Using Hifi-Gan model")
                        # Clear hifigan mel directories
                        if os.path.exists("hifi-gan/test_mel_files"):
                            shutil.rmtree("hifi-gan/test_mel_files")
                        if os.path.exists("hifi-gan/generated_files_from_mel"):
                            shutil.rmtree("hifi-gan/generated_files_from_mel")

                        os.makedirs("hifi-gan/test_mel_files")

                        text = p
                        sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
                        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
                        mel_outputs, mel_outputs_postnet, _, alignments = self.taco_model.inference(sequence)
                        mel_out = mel_outputs_postnet.cpu().detach().numpy()
                        np.save("hifi-gan/test_mel_files/mel.data", mel_out)

                        tp = subprocess.run(['python', '-u', 'hifi-gan/inference_e2e.py', '--checkpoint_file', self.hifigan_model_path, '--output_dir', 'hifi-gan/generated_files_from_mel', '--input_mels_dir', 'hifi-gan/test_mel_files'])
                        #copy file
                        wav_name =  dpg.get_value("infer_project_name") + "/wavs_out/" + str(self.file_count) + ".wav"
                        shutil.move("hifi-gan/generated_files_from_mel/mel.data_generated_e2e.wav", wav_name)
                        inferer.file_count += 1
                        result.append([text, wav_name])


                    elif chosen_model == "Use Waveglow model":
                        text = p
                        sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
                        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
                        mel_outputs, mel_outputs_postnet, _, alignments = self.taco_model.inference(sequence)

                        self.waveglow_model = torch.load(self.waveglow_model_path)['model']
                        self.waveglow_model.cuda().eval().half()
                        for k in self.waveglow_model.convinv:
                            k.float()
                        self.denoiser = Denoiser(self.waveglow_model) 

                        with torch.no_grad():
                            audio = self.waveglow_model.infer(mel_outputs_postnet, sigma=1)
                        audio_denoised = self.denoiser(audio, strength=0.02)[:, 0]
                        audioout = audio_denoised[0].data.cpu().numpy()
                        #audioout = audio[0].data.cpu().numpy()
                        audioout32 = np.float32(audioout)
    
                        wav_name = dpg.get_value("infer_project_name")+ '/wavs_out/' + str(self.file_count) + '.wav'
                        result.append([text, wav_name])
                        sf.write(wav_name, audioout32, 22050)
                        inferer.file_count += 1

                return result

def callback_infer_open_model_taco(sender, app_data):
    path = app_data["file_path_name"]
    path = path.rstrip('.*')
    inferer.set_taco_model_path(path)
    status = dpg.get_value("infer_status_text")
    dpg.set_value("infer_status_text", status + "\nTacotron2 model {} selected.".format(path))
    dpg.set_y_scroll("infer_status_window", 1000)

def callback_infer_open_model_hifigan(sender, app_data):
    path = app_data["file_path_name"]
    path = path.rstrip('.*')
    inferer.set_hifigan_model_path(path)
    status = dpg.get_value("infer_status_text")
    dpg.set_value("infer_status_text", status + "\nHifi-Gan Model {} selected.".format(path))
    dpg.set_y_scroll("infer_status_window", 1000)    

def callback_infer_open_model_waveglow(sender, app_data):
    path = app_data["file_path_name"]
    path = path.rstrip('.*')
    inferer.set_waveglow_model_path(path)
    status = dpg.get_value("infer_status_text")
    dpg.set_value("infer_status_text", status + "\nWaveglow Model {} selected.".format(path))
    dpg.set_y_scroll("infer_status_window", 1000)  

def callback_infer_open_text_file(sender, app_data):
    d_path = app_data["selections"]
    key = list(d_path.keys())[0]
    path = app_data["selections"][key]
    inferer.set_text_file_path(path)
    status = dpg.get_value("infer_status_text")
    dpg.set_value("infer_status_text", status + "\nText file {} selected.".format(path))
    dpg.set_y_scroll("infer_status_window", 1000)    

def callback_open_project(sender, app_data):
    path = app_data["file_path_name"]
    path = path.rstrip('.*')
    trainer.set_hifigan_project_name(path)

def callback_open_project_checkpoint(sender, app_data):
    path = app_data["file_path_name"]
    path = path.rstrip('.*')
    print("PATH {}".format(path))
    trainer.set_hifigan_checkpoint_name(path)

def callback_run_inference(sender, app_data, user_data):
    # check to see if models selected
    if not inferer.get_hifigan_model_path() and dpg.get_value("infer_model_radio") == "Use Hifi-Gan model":
        status = dpg.get_value("infer_status_text")
        dpg.set_value("infer_status_text", status + "\nError: Hifi-Gan mode but no model chosen!")
        dpg.set_y_scroll("infer_status_window", 1000)
        return
    if not inferer.get_waveglow_model_path() and dpg.get_value("infer_model_radio") == "Use Waveglow model":
        status = dpg.get_value("infer_status_text")
        dpg.set_value("infer_status_text", status + "\nError: Waveglow mode but no model chosen!")
        dpg.set_y_scroll("infer_status_window", 1000)            
        return
    if not inferer.get_taco_model_path():
        status = dpg.get_value("infer_status_text")
        dpg.set_value("infer_status_text", status + "\nError: Tacotron2 model not chosen!")
        dpg.set_y_scroll("infer_status_window", 1000)
        return

    if user_data == "single": 
        t = dpg.get_value("text_input")
        if t:
            print("running inference")            
            status = dpg.get_value("infer_status_text")
            dpg.set_value("infer_status_text", status + "\nRunning inference...")
            dpg.set_y_scroll("infer_status_window", 1000)

            result = inferer.run_inference(t, "text_input", None)
            a = AudioSegment.from_file(result[0]) 
            t_play = threading.Thread(target=inferer.play, args=(a,))
            t_play.start()
            entry = np.array([t, result[0]])
            inferer.add_entry([entry])
            status = dpg.get_value("infer_status_text")
            dpg.set_value("infer_status_text", status + "\nInferred entry added to table.")
            dpg.set_y_scroll("infer_status_window", 1000)
        else:
            print("Nothing to infer!")
            status = dpg.get_value("infer_status_text")
            dpg.set_value("infer_status_text", status + "\nError: no input to infer!")
            dpg.set_y_scroll("infer_status_window", 1000)
            return
    
    elif user_data == "file":
        print(inferer.get_text_file_path())
        if inferer.get_text_file_path():
            if os.path.exists(inferer.get_text_file_path()):
                status = dpg.get_value("infer_status_text")
                dpg.set_value("infer_status_text", status + "\nRunning inference...")
                dpg.set_y_scroll("infer_status_window", 1000)
                with open(inferer.get_text_file_path(), 'r') as f:
                    r = f.readlines()
                    text_file = " ".join(r)
                    result = inferer.run_inference(text_file, "text_input_file", None)
                    for r in result:
                        entry = np.array([r[0], r[1]])
                        inferer.add_entry([entry])
                status = dpg.get_value("infer_status_text")
                dpg.set_value("infer_status_text", status + "\nInferring text file completed.")
                dpg.set_y_scroll("infer_status_window", 1000)
        else:
            status = dpg.get_value("infer_status_text")
            dpg.set_value("infer_status_text", status + "\nError: Text file not chosen!")
            dpg.set_y_scroll("infer_status_window", 1000)
            return



def callback_train_taco(sender, data):
    print("running taco training")
    trainer.train_taco()

def callback_train_hifigan(sender, data):
    if trainer.is_running():
        return
    print("running hifigan training")
    trainer.train_hifigan()

def callback_stop_training(sender, data):
    trainer.stop_training_hifigan()
    trainer.stop_training_tacotron2()
    trainer.stop_training_waveglow()

def callback_start_tensorboard(sender, data):
    path = trainer.get_hifigan_checkpoint_name()
    if not path:
        return
    import webbrowser as web
    web.open("http://localhost:6006")
    # out = "tensorboard --logdir '/model utilities/hifi-gan/attenborough.model/logs'"
    # print(out)
    tp = subprocess.Popen(['tensorboard', '--logdir', path])
    trainer.set_tensorboard_process(tp)

def callback_export_infer_table(sender, data):
    if inferer.get_table_array().size > 0:
        p_path = dpg.get_value("infer_project_name")
        with open(p_path + "/wav_list.csv", 'w') as f:
            t = dpg.get_item_children("infer_table")
            for i in range(0, len(t[1])):
                f.write(dpg.get_value("wav_path_" + str(i)))
                f.write('|')
                f.write(dpg.get_value("input_text_" + str(i)))
                f.write('\n')
        status = dpg.get_value("infer_status_text")
        dpg.set_value("infer_status_text", status + "\nTable exported to project name")
        dpg.set_y_scroll("infer_status_window", 1000)
    else:
        status = dpg.get_value("infer_status_text")
        dpg.set_value("infer_status_text", status + "\nError: no table to export!")
        dpg.set_y_scroll("infer_status_window", 1000)

def callback_status_window_control(sender, data):
    if dpg.is_item_active("inference_tab"):
        dpg.configure_item("infer_status_window", show=True)
        dpg.configure_item("infer_table_window", show=True)
        dpg.configure_item("infer_text_file_window", show=True)
        dpg.configure_item("infer_single_text_window", show=True)
        dpg.configure_item("infer_choose_model_window", show=True)
        
    elif dpg.is_item_active("train_hifigan_tab"):
        dpg.configure_item("infer_status_window", show=False)
        dpg.configure_item("infer_table_window", show=False)
        dpg.configure_item("infer_text_file_window", show=False)
        dpg.configure_item("infer_single_text_window", show=False)
        dpg.configure_item("infer_choose_model_window", show=False)        

    #hide other tab windows

with dpg.window(tag='mainwindow', label="Model Utilites"):
   
    with dpg.tab_bar(tag="tab_bar_1"):        
            
        with dpg.tab(tag="inference_tab", label=" Run Inference "):


            with dpg.file_dialog(modal=True, width=800, directory_selector=False, show=False, callback=callback_infer_open_model_taco, tag="infer_open_model_taco"):
                dpg.add_file_extension(".*", color=(255, 255, 255, 255))

            with dpg.file_dialog(modal=True, width=800, directory_selector=False, show=False, callback=callback_infer_open_model_hifigan, tag="infer_open_model_hifigan"):
                dpg.add_file_extension(".*", color=(255, 255, 255, 255))

            with dpg.file_dialog(modal=True, width=800, directory_selector=False, show=False, callback=callback_infer_open_model_waveglow, tag="infer_open_model_waveglow"):
                dpg.add_file_extension(".*", color=(255, 255, 255, 255))

            with dpg.file_dialog(modal=True, width=800, directory_selector=False, show=False, callback=callback_infer_open_text_file, tag="open_text_file"):
                dpg.add_file_extension(".*", color=(255, 255, 255, 255))

            with dpg.file_dialog(modal=True, width=800, directory_selector=True, show=False, callback=callback_open_project, tag="open_project_dialog"):
                dpg.add_file_extension(".*", color=(255, 255, 255, 255))                

            with dpg.file_dialog(modal=True, width=800, directory_selector=True, show=False, callback=callback_open_project_checkpoint, tag="open_project_checkpoint_dialog"):
                dpg.add_file_extension(".*", color=(255, 255, 255, 255))   

            with dpg.window(tag="infer_status_window", show=True, width=600, height=160, pos=(535,35), horizontal_scrollbar=True, menubar=False, no_resize=True, no_title_bar=True, no_move=True, no_scrollbar=False, no_collapse=True, no_close=True):
                dpg.add_text("Status...", tag="infer_status_text")            

            with dpg.window(tag="infer_choose_model_window", show=True, width=525, height=160, pos=(5,35), menubar=False, no_resize=True, no_title_bar=True, no_move=True, no_scrollbar=True, no_collapse=True, no_close=True):     
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Project name: ")
                    dpg.add_input_text(default_value="MyProject", tag="infer_project_name")    
                dpg.add_spacer(height=3)
                with dpg.group(horizontal=True):
                    dpg.add_text("Produce audio from nvidia tacotron2 model:")
                    dpg.add_button(label="Choose Tacotron2 model", tag="choose_model_taco", callback=lambda: dpg.show_item("infer_open_model_taco"))
                dpg.add_spacer(height=3)                
                with dpg.group(horizontal=True):
                    dpg.add_radio_button(items=["Use Hifi-Gan model", "Use Waveglow model"], tag="infer_model_radio", default_value="Use Hifi-Gan model", horizontal=False)
                    with dpg.group():
                        dpg.add_button(label="Choose model", tag="infer_open_hifigan_model", callback=lambda: dpg.show_item("infer_open_model_hifigan"))
                        dpg.add_button(label="Choose model", tag="infer_open_waveglow_model", callback=lambda: dpg.show_item("infer_open_model_waveglow"))

             
            with dpg.window(tag="infer_single_text_window", show=True, width=850, height=60, pos=(5,200), menubar=False, no_resize=True, no_title_bar=True, no_move=True, no_scrollbar=True, no_collapse=True, no_close=True):     
                dpg.add_text("Input single line text:")             
                dpg.add_input_text(width=800, tag="text_input")
                dpg.add_button(label="Run inference", tag="run_inference_single", callback=callback_run_inference, user_data="single")
            with dpg.window(tag="infer_text_file_window", show=True, width=250, height=100, pos=(860,200), menubar=False, no_resize=True, no_title_bar=True, no_move=True, no_scrollbar=True, no_collapse=True, no_close=True):     
                dpg.add_text("Input text file:")
                with dpg.group(horizontal=True):              
                    dpg.add_button(label="Choose text file", tag="choose_text_file", callback=lambda: dpg.show_item("open_text_file"))
                    dpg.add_button(label="Run inference", tag="run_inference", callback=callback_run_inference, user_data="file")

            with dpg.window(tag="infer_table_window", no_background=True, show=True, width=1350, height=440, pos=(5,310), menubar=False, no_resize=True, no_title_bar=True, no_move=True, no_scrollbar=True, no_collapse=True, no_close=True):     
                    dpg.add_button(label="Export .csv", tag="export_infer_table", callback=callback_export_infer_table)

        # with dpg.tab(tag="train_tacotron2_tab", label=" Train Tacotron2 "):
        #     dpg.add_spacer()    
        #     dpg.add_button(label="Train Tacotron2 model", tag="train_taco")
        #     dpg.add_clicked_handler("train_taco", callback=callback_train_taco)
        #     dpg.add_spacer()    
        #     dpg.add_button(label="Stop training", tag="stop_training_tacotron2")
        #     dpg.add_clicked_handler("stop_training_tacotron2", callback=callback_stop_training)       
        #     dpg.add_spacer()    
        #     dpg.add_text("Shell output displayed here", tag="shell_output_tacotron2")      

        with dpg.tab(tag="train_hifigan_tab", label=" Train Hifi-Gan "):  
            dpg.add_spacer()    
            dpg.add_text("Project folder should contain audio clips in its /wavs directory.\nTraining file should be named 'training.csv' and validation file named 'validation.csv'")
            dpg.add_spacer()                
            dpg.add_button(label="Choose project folder", tag="open_project", callback=lambda: dpg.show_item("open_project_dialog"))
            dpg.add_spacer()                
            dpg.add_button(label="Choose checkpoint folder (none for new)", tag="open_project_checkpoint", callback=lambda: dpg.show_item("open_project_checkpoint_dialog"))
            dpg.add_spacer()    
            dpg.add_button(label="Train HifiGan model", tag="train_hifigan", callback=callback_train_hifigan)
            dpg.add_spacer()    
            dpg.add_button(label="Stop training", tag="stop_training_hifigan", callback=callback_stop_training)
            dpg.add_spacer()    
            dpg.add_button(label="Start Tensorboard", tag="start_tensorboard", callback=callback_start_tensorboard)
            dpg.add_spacer()    
            dpg.add_text("Shell output displayed here", tag="shell_output_hifigan")  

        # with dpg.tab(tag="train_waveglow_tab", label=" Train Waveglow "):        
        #     dpg.add_spacer()    
        #     dpg.add_button(label="Stop training", tag="stop_training_waveglow")
        #     dpg.add_clicked_handler("stop_training_waveglow", callback=callback_stop_training)       
        #     dpg.add_spacer()    
        #     dpg.add_text("Shell output displayed here", tag="shell_output_waveglow")

inferer = Inferer()
trainer = Trainer()
   

with dpg.item_handler_registry(tag="status_window_handler"):
    dpg.add_item_active_handler(callback=callback_status_window_control)
dpg.bind_item_handler_registry("train_hifigan_tab", "status_window_handler")
dpg.bind_item_handler_registry("inference_tab", "status_window_handler")

with dpg.font_registry():
    default_font = dpg.add_font("CheyenneSans-Light.otf", 17)
    font2 = dpg.add_font("PublicSans-Regular.otf", 18)
    font3 = dpg.add_font("VarelaRound-Regular.ttf", 17)
    

dpg.bind_font(font2)

with dpg.theme() as global_theme:

    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (7, 18, 54), category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)

    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (59, 58, 68), category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)

    # with dpg.theme_component(dpg.mvInputInt):
    #     dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (140, 255, 23), category=dpg.mvThemeCat_Core)
    #     dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)

dpg.bind_theme(global_theme)

dpg.create_viewport(title="Deep Voice Model Utilities v1.0 by YouMeBangBang", width=1400, height=800)
dpg.setup_dearpygui()
dpg.show_viewport()

dpg.set_global_font_scale(1.0)

dpg.set_primary_window("mainwindow", True)
dpg.start_dearpygui()
dpg.destroy_context()


