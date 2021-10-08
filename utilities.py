import dearpygui.dearpygui as dpg
import os

#Tacotron imports
import os
import librosa
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

class Trainer():
    def __init__(self):
        self.process = None
        self.is_training_running = False
        self.t1 = None
        self.t2 = None      

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
            self.process =  subprocess.Popen(['python', '-u', 'train.py', '--checkpoint_path', 'attenborough.model',
            '--input_training_file', 'training.csv', '--input_validation_file', 'validation.csv', '--input_wavs_dir', 'wavs'] , stdout=subprocess.PIPE)  
           
            # self.process =  subprocess.run(['python', '-u', 'train.py', '--checkpoint_path', 'attenborough.model',
            # '--input_training_file', 'training.csv', '--input_validation_file', 'validation.csv', '--input_wavs_dir', 'wavs'] , capture_output=True, text=True)  
            
            while self.is_training_running:
                time.sleep(.01)
                out = self.process.stdout.readline().decode('utf-8') 
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
            os.chdir("../")

        

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
        self.taco_model_name = None
        self.taco_model_path = None
        self.waveglow_model_name = None
        self.waveglow_model_path = None
        self.hifigan_model_name = None
        self.hifigan_model_path = None
        self.taco_model = None
        self.waveglow_model = None
        self.hifigan_model = None
        self.denoiser = None
        self.text_file_path = None
        self.project_path = None
    
    def set_taco_model_path(self, path):
        self.taco_model_path = path
    def set_waveglow_model_path(self, path):
        self.waveglow_model_path = path
    def set_hifigan_model_path(self, path):
        self.hifigan_model_path = path
    def set_text_file_path(self, path):
        self.text_file_path = path
    def set_project_path(self, path):
        self.project_path = path    

    def get_text_file_path(self):
        return self.text_file_path

    def run_inference(self, input_text, mode):
        hparams = create_hparams()
        hparams.sampling_rate = 22050
        #hparams change dropouts!  
        hparams.p_attention_dropout = 0
        hparams.p_decoder_dropout = 0
        hparams.max_decoder_steps = 10000
   
        self.taco_model = load_model(hparams)
        self.taco_model.load_state_dict(torch.load(self.taco_model_path)['state_dict'])
        _ = self.taco_model.cuda().eval().half()
        self.waveglow_model = torch.load(self.waveglow_model_path)['model']
        self.waveglow_model.cuda().eval().half()
        for k in self.waveglow_model.convinv:
            k.float()
        self.denoiser = Denoiser(self.waveglow_model)
        #text = "a log file will be created when first opening a project. The last entry that was edited is recorded, so that work can easily be resumed."
        if mode == "text_input":
            text = input_text
            sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
            sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
            mel_outputs, mel_outputs_postnet, _, alignments = self.taco_model.inference(sequence)
            
            with torch.no_grad():
                audio = self.waveglow_model.infer(mel_outputs_postnet, sigma=1)
            audio_denoised = self.denoiser(audio, strength=0.02)[:, 0]
            audioout = audio_denoised[0].data.cpu().numpy()
            #audioout = audio[0].data.cpu().numpy()
            audioout32 = np.float32(audioout)    
            sf.write('out.wav', audioout32, 22050)
            a = AudioSegment.from_file("out.wav") 
            play(a)

        elif mode == "text_input_file":
            #break text apart and infer each phrase.
            #get max word length of phrase if no punctuation
            import re
            phrase_splits = re.split(r'(?<=[\.\!\?])\s*', input_text)   #split on white space between sentences             
            phrase_splits = list(filter(None, phrase_splits))  #remove empty splits
            if phrase_splits:
                audio_list = []
                print(phrase_splits)

                for i, p in enumerate(phrase_splits):
                    text = p
                    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
                    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
                    mel_outputs, mel_outputs_postnet, _, alignments = self.taco_model.inference(sequence)
                    
                    with torch.no_grad():
                        audio = self.waveglow_model.infer(mel_outputs_postnet, sigma=1)
                    audio_denoised = self.denoiser(audio, strength=0.02)[:, 0]
                    audioout = audio_denoised[0].data.cpu().numpy()
                    #audioout = audio[0].data.cpu().numpy()
                    audioout32 = np.float32(audioout)
                    audio_list.append(audioout32)    
                    sf.write('out' + str(i) + '.wav', audioout32, 22050)
                    a = AudioSegment.from_file("out" + str(i) + ".wav") 
                    play(a)




inferer = Inferer()
trainer = Trainer()

def setup_install_reqs(sender, appdata, data):
    print("installing reqs...")
    os.system("pip install -r requirements.txt --user")

def callback_open_model_taco(sender, app_data):
    print(app_data["file_name_buffer"])
    print(app_data["current_path"])
    path = app_data["current_path"] + '/' + app_data["file_name_buffer"]
    inferer.set_taco_model_path(path)

def callback_open_model_waveglow(sender, app_data):
    print(app_data["file_name_buffer"])
    print(app_data["current_path"])
    path = app_data["current_path"] + '/' + app_data["file_name_buffer"]
    inferer.set_waveglow_model_path(path)

def callback_open_model_hifigan(sender, app_data):
    print(app_data["file_name_buffer"])
    print(app_data["current_path"])
    path = app_data["current_path"] + '/' + app_data["file_name_buffer"]
    inferer.set_hifigan_model_path(path)

def callback_open_text_file(sender, app_data):
    print(app_data["file_name_buffer"])
    print(app_data["current_path"])
    path = app_data["current_path"] + '/' + app_data["file_name_buffer"]
    inferer.set_text_file_path(path)

def callback_open_project(sender, app_data):
    print(app_data["file_name_buffer"])
    print(app_data["current_path"])
    path = app_data["current_path"] + '/' + app_data["file_name_buffer"]
    inferer.set_project_path(path)

def callback_run_inference(sender, app_data): 
    t = dpg.get_value("text_input")
    text_file = None
    if inferer.get_text_file_path():
        if os.path.exists(inferer.get_text_file_path()):
            with open(inferer.get_text_file_path(), 'r') as f:
                r = f.readlines()
                text_file = " ".join(r)
                print("opened text file.")
    if t:
        print("running inference")
        inferer.run_inference(t, "text_input")
    elif text_file:
        print("running inference")
        inferer.run_inference(text_file, "text_input_file")

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

vp = dpg.create_viewport(title="Deep Voice Model Utilities v1.0 by YouMeBangBang", width=1200, height=800)

with dpg.window(id='mainwindow', label="Model Utilites"):
   
    with dpg.tab_bar(id="tab_bar_1"):        
        with dpg.tab(id="setup_tab", label=" Setup and Config "):
            dpg.add_spacing(count=5)
            dpg.add_text("System setup:")
            dpg.add_button(label="Install requirements", callback=setup_install_reqs)
            
        with dpg.tab(id="inference_tab", label=" Run Inference "):
            with dpg.file_dialog(modal=True, width=600, directory_selector=False, show=False, callback=callback_open_model_taco, id="open_model_taco"):
                dpg.add_file_extension(".*", color=(255, 255, 255, 255))

            with dpg.file_dialog(modal=True, width=600, directory_selector=False, show=False, callback=callback_open_model_waveglow, id="open_model_waveglow"):
                dpg.add_file_extension(".*", color=(255, 255, 255, 255))

            with dpg.file_dialog(modal=True, width=600, directory_selector=False, show=False, callback=callback_open_model_hifigan, id="open_model_hifigan"):
                dpg.add_file_extension(".*", color=(255, 255, 255, 255))

            with dpg.file_dialog(modal=True, width=600, directory_selector=False, show=False, callback=callback_open_text_file, id="open_text_file"):
                dpg.add_file_extension(".*", color=(255, 255, 255, 255))

            with dpg.file_dialog(modal=True, width=600, directory_selector=True, show=False, callback=callback_open_project, id="open_project_dialog"):
                dpg.add_file_extension(".*", color=(255, 255, 255, 255))                

            with dpg.table(label="Inferred wavs", id="infer_table", pos=([500,100]), height=600, width=600):
                dpg.add_table_column()
                dpg.add_table_column()
                dpg.add_table_column()  
            dpg.add_spacing(count=5)
            dpg.add_text("Produce audio from nvidia tacotron2 model:")
            dpg.add_spacing(count=5)
            dpg.add_button(label="Choose Tacotron2 model", id="choose_model_taco")
            dpg.add_clicked_handler("choose_model_taco", callback=lambda: dpg.show_item("open_model_taco"))
            dpg.add_same_line(spacing=10)
            dpg.add_text("", id="tacotron2_model_status")
            dpg.add_spacing(count=5)
            dpg.add_button(label="Choose Hifi-Gan model", id="choose_model_hifigan")  
            dpg.add_clicked_handler("choose_model_hifigan", callback=lambda: dpg.show_item("open_model_hifigan"))  
            dpg.add_same_line(spacing=10)
            dpg.add_text("", id="hifigan_model_status")            
            dpg.add_same_line(spacing=10)   
            dpg.add_text("Or")       
            dpg.add_same_line(spacing=10)     
            dpg.add_button(label="Choose Waveglow model", id="choose_model_waveglow")
            dpg.add_clicked_handler("choose_model_waveglow", callback=lambda: dpg.show_item("open_model_waveglow"))   
            dpg.add_same_line(spacing=10)
            dpg.add_text("", id="waveglow_model_status")                 
            dpg.add_spacing(count=5)    
            dpg.add_text("Input text:")
            dpg.add_input_text(width=800, id="text_input")
            dpg.add_spacing(count=5)    
            dpg.add_button(label="Choose text file", id="choose_text_file")
            dpg.add_clicked_handler("choose_text_file", callback=lambda: dpg.show_item("open_text_file"))              
            dpg.add_spacing(count=5)    
            dpg.add_button(label="run inference", id="run_inference")
            dpg.add_clicked_handler("run_inference", callback=callback_run_inference)
        
        with dpg.tab(id="train_tacotron2_tab", label=" Train Tacotron2 "):
            dpg.add_spacing(count=5)    
            dpg.add_button(label="Train Tacotron2 model", id="train_taco")
            dpg.add_clicked_handler("train_taco", callback=callback_train_taco)
            dpg.add_spacing(count=5)    
            dpg.add_button(label="Stop training", id="stop_training_tacotron2")
            dpg.add_clicked_handler("stop_training_tacotron2", callback=callback_stop_training)       
            dpg.add_spacing(count=5)    
            dpg.add_text("Shell output displayed here", id="shell_output_tacotron2")      

        with dpg.tab(id="train_hifigan_tab", label=" Train Hifi-Gan "):  
            dpg.add_spacing(count=5)    
            dpg.add_text("Project folder should contain audio clips in its /wavs directory.\nTraining file should be named 'training.csv' and validation file named 'validation.csv'")
            dpg.add_spacing(count=5)                
            dpg.add_button(label="Choose project folder", id="open_project")
            dpg.add_clicked_handler("open_project", callback=lambda: dpg.show_item("open_project_dialog"))             
            dpg.add_spacing(count=5)    
            dpg.add_button(label="Train HifiGan model", id="train_hifigan")
            dpg.add_clicked_handler("train_hifigan", callback=callback_train_hifigan)  
            dpg.add_spacing(count=5)    
            dpg.add_button(label="Stop training", id="stop_training_hifigan")
            dpg.add_clicked_handler("stop_training_hifigan", callback=callback_stop_training)       
            dpg.add_spacing(count=5)    
            dpg.add_text("Shell output displayed here", id="shell_output_hifigan")  

        with dpg.tab(id="train_waveglow_tab", label=" Train Waveglow "):        
            dpg.add_spacing(count=5)    
            dpg.add_button(label="Stop training", id="stop_training_waveglow")
            dpg.add_clicked_handler("stop_training_waveglow", callback=callback_stop_training)       
            dpg.add_spacing(count=5)    
            dpg.add_text("Shell output displayed here", id="shell_output_waveglow")


dpg.setup_dearpygui(viewport=vp)
dpg.show_viewport(vp)

dpg.set_global_font_scale(1.3)
dpg.set_primary_window("mainwindow", True)


dpg.start_dearpygui()

