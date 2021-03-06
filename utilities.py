import psutil
import os
import shutil
import dearpygui.dearpygui as dpg
from dearpygui.dearpygui import get_value
import sys
import numpy as np
import torch
sys.path.append('tacotron2/')
sys.path.append('waveglow/')
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
import json

dpg.create_context()


class Trainer():
    def __init__(self):
        self.train_process = None
        self.is_training_running = False
        self.project_folder = False
        self.model_type = None
        self.is_training_paused = False

        self.training_thread = None


        self.taco_checkpoint_path = None
        self.hifigan_checkpoint_path = None
        self.waveglow_checkpoint_path = None

 
        self.tensorboard_process = None        
    
    def set_tensorboard_process(self, p):
        self.tensorboard_process = p

    def get_tensorboard_process(self):
        return self.tensorboard_process

    def set_taco_checkpoint_path(self, path):
        self.taco_checkpoint_path = path    

    def get_taco_checkpoint_path(self):
        return self.taco_checkpoint_path

    def set_hifigan_checkpoint_path(self, path):
        self.hifigan_checkpoint_path = path    

    def get_hifigan_checkpoint_path(self):
        return self.hifigan_checkpoint_path

    def set_waveglow_checkpoint_path(self, path):
        self.waveglow_checkpoint_path = path    

    def get_waveglow_checkpoint_path(self):
        return self.waveglow_checkpoint_path        

    def set_project_folder(self, path):
        self.project_folder = path    

    def get_project_folder(self):
        return self.project_folder

    def is_running(self):
        return self.is_training_running

    def is_paused(self):
        return self.is_training_paused

    def train_model(self, model_type, batch_size, iters_per_checkpoint, learning_rate, multigpu):

        self.is_training_running = True
     
        def train_hifigan():
            os.chdir("hifi-gan") 

            # Save parameters to hifigan json config_v3
            with open("config_v3.json", "r") as f:
                data = json.load(f)
                data['batch_size'] = int(batch_size)
                data['learning_rate'] = float(learning_rate)                
                # if multigpu:
                #     data['num_gpus'] = 0
            with open("config_v3.json", 'w') as f:
                json.dump(data, f, indent=4)


            if self.hifigan_checkpoint_path:
                self.train_process =  subprocess.Popen(['python', '-u', 'train.py', '--checkpoint_path', self.hifigan_checkpoint_path, '--checkpoint_interval', iters_per_checkpoint,
                '--input_training_file', self.project_folder + '/' + 'training.csv', '--input_validation_file', self.project_folder + '/' + 'validation.csv', '--input_wavs_dir', self.project_folder + '/' + 'wavs'] , stdout=subprocess.PIPE)  
            else:
                #new checkpoint
                self.train_process =  subprocess.Popen(['python', '-u', 'train.py', '--checkpoint_path', self.project_folder + '_model', '--checkpoint_interval', iters_per_checkpoint,
                '--input_training_file', self.project_folder + '/' + 'training.csv', '--input_validation_file', self.project_folder + '/' + 'validation.csv', '--input_wavs_dir', self.project_folder + '/' + 'wavs'] , stdout=subprocess.PIPE)  

            os.chdir("../")
            while self.is_training_running:
                time.sleep(.01)
                out = self.train_process.stdout.readline().decode('utf-8')
                if out: 
                    print(out)
                    dpg.set_value("trainer_status_output", dpg.get_value("trainer_status_output") + "\n" + out)
                    dpg.set_y_scroll("trainer_status_window", 1000000)              

        def train_taco():
            os.chdir("tacotron2") 
            os.makedirs(self.project_folder + '/checkpoints/logs', exist_ok=True)

            if multigpu:
                run_command = ['python', '-u', 'multiproc.py']
            else:
                run_command = ['python', '-u', 'train.py']         
                
            if self.taco_checkpoint_path:
                run_command.extend(['--checkpoint_path', self.taco_checkpoint_path])

            if multigpu:
                run_command.extend(['--hparams', 'batch_size={}, iters_per_checkpoint={}, learning_rate={}, training_files={}, validation_files={}, project_path={}, distributed_run=True'.format(int(batch_size), int(iters_per_checkpoint), float(learning_rate), self.project_folder + '/' + 'training.csv', self.project_folder + '/' + 'validation.csv', self.project_folder)])
            else:
                run_command.extend(['--hparams', 'batch_size={}, iters_per_checkpoint={}, learning_rate={}, training_files={}, validation_files={}, project_path={}, distributed_run=False'.format(int(batch_size), int(iters_per_checkpoint), float(learning_rate), self.project_folder + '/' + 'training.csv', self.project_folder + '/' + 'validation.csv', self.project_folder)])
      
            run_command.extend(['--log_directory', 'logs' ])
            run_command.extend(['--output_directory', self.project_folder + '/checkpoints' ])

            self.train_process =  subprocess.Popen(run_command, stdout=subprocess.PIPE)  
           
            os.chdir("../")

            while self.is_training_running:
                time.sleep(.01)
                out = self.train_process.stdout.readline().decode('utf-8')
                if out: 
                    print(out)
                    dpg.set_value("trainer_status_output", dpg.get_value("trainer_status_output") + "\n" + out)
                    dpg.set_y_scroll("trainer_status_window", 1000000)              

        def train_waveglow():
            os.chdir("waveglow") 

            # Save parameters to waveglow json config
            with open("config.json", "r") as f:
                data = json.load(f)
                data['train_config']['project_folder'] = self.project_folder
                data['train_config']['fp16_run'] = False
                data['train_config']['with_tensorboard'] = True                       
                data['train_config']['batch_size'] = int(batch_size)
                data['train_config']['learning_rate'] = float(learning_rate)
                data['train_config']['output_directory'] = self.project_folder + '/' + "checkpoints"
                data['train_config']['iters_per_checkpoint'] = int(iters_per_checkpoint)
                data['data_config']['training_files'] = self.project_folder + '/' + "training.csv"
                if self.waveglow_checkpoint_path:
                    data['train_config']['checkpoint_path'] = self.waveglow_checkpoint_path
            with open("config.json", 'w') as f:
                json.dump(data, f, indent=4)

            if multigpu:
                self.train_process =  subprocess.Popen(['python', '-u', 'distributed.py', '--config', 'config.json'], stdout=subprocess.PIPE)  
            else:
                self.train_process =  subprocess.Popen(['python', '-u', 'train.py', '--config', 'config.json'], stdout=subprocess.PIPE)  

            os.chdir("../")

            while self.is_training_running:
                time.sleep(.01)
                out = self.train_process.stdout.readline().decode('utf-8')
                if out: 
                    print(out)
                    dpg.set_value("trainer_status_output", dpg.get_value("trainer_status_output") + "\n" + out)
                    dpg.set_y_scroll("trainer_status_window", 1000000)              

        self.model_type = model_type
        if (model_type == "Train Hifi-Gan model"):
            self.training_thread = threading.Thread(target=train_hifigan)
            self.training_thread.start()
            print("HifiGan training started!")
            dpg.set_value("trainer_status_output", dpg.get_value("trainer_status_output") + "\nHifi-Gan training started.")
            dpg.set_y_scroll("trainer_status_window", 1000000)              

        if (model_type == "Train Tacotron2 model"):
            self.training_thread = threading.Thread(target=train_taco)
            self.training_thread.start()
            print("Tacotron2 training started!")
            dpg.set_value("trainer_status_output", dpg.get_value("trainer_status_output") + "\nTacotron2 training started.")
            dpg.set_y_scroll("trainer_status_window", 1000000)              

        if (model_type == "Train Waveglow model"):
            self.training_thread = threading.Thread(target=train_waveglow)
            self.training_thread.start()
            print("Waveglow training started!")
            dpg.set_value("trainer_status_output", dpg.get_value("trainer_status_output") + "\nWaveglow training started.")
            dpg.set_y_scroll("trainer_status_window", 1000000)              

    def stop_training(self):
        print("\nStopping training...")
        if self.train_process:
            self.train_process.kill()
            time.sleep(1)
            self.is_training_running = False
            dpg.set_value("trainer_status_output", dpg.get_value("trainer_status_output") + "\nTraining process stopped.")
            dpg.set_y_scroll("trainer_status_window", 1000000)              
            print("Training stopped successfully.")

    def pause_training(self):
        if self.is_training_paused:
            # resume training
            self.is_training_paused = False
            psProcess = psutil.Process(pid=self.train_process.pid)
            psProcess.resume()
            dpg.configure_item("trainer_pause_training", label="Pause training")
            dpg.set_value("trainer_status_output", dpg.get_value("trainer_status_output") + "\nTraining process resumed.")
            dpg.set_y_scroll("trainer_status_window", 1000000)                   
        else:
            self.is_training_paused = True
            psProcess = psutil.Process(pid=self.train_process.pid)
            psProcess.suspend()
            dpg.configure_item("trainer_pause_training", label="Resume training")
            dpg.set_value("trainer_status_output", dpg.get_value("trainer_status_output") + "\nTraining process paused.")
            dpg.set_y_scroll("trainer_status_window", 1000000)              
       


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
        self.id_tag = 0
        self.table_array = np.empty([0,2])
        self.is_running = False
  
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

    def get_is_running(self):
        return self.is_running

    def show_table(self):
        # clear table      
        dpg.delete_item("infer_table")
        if dpg.does_alias_exist("infer_table"):      
            dpg.delete_alias("infer_table")

        with dpg.table(pos=(0, 0), resizable=False, scrollY=True, row_background=True, borders_innerH=True, borders_outerH=True, borders_innerV=True,
                borders_outerV=True, parent="infer_table_window", header_row=True, width=1325, height=400, tag="infer_table") as infer_table_object:
                
            dpg.add_table_column(width_fixed=True, init_width_or_weight=875, label='TEXT')
            dpg.add_table_column(width_fixed=True, init_width_or_weight=250, label='AUDIO FILE')
            dpg.add_table_column(width_fixed=True, init_width_or_weight=200, label='OPTIONS')  

            l = len(self.table_array)
            for i in range(0, l): 
                with dpg.table_row(): 
                    dpg.add_input_text(tag="input_text_" + str(self.id_tag), default_value=str(self.table_array[i][0]), width=850)
                    dpg.add_text(str(self.table_array[i][1]))

                    a_path = str(self.table_array[i][1])
                    entry_info = {
                        "rank": i,
                        "text": dpg.get_value("input_text_" + str(i)),
                        "wav_path": a_path,
                        "tag_name": "input_text_" + str(self.id_tag)
                    }
                    self.id_tag += 1

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
        a_length = a.duration_seconds
        status = dpg.get_value("infer_status_text")
        dpg.set_value("infer_status_text", status + "\nPlaying audio with length {0:.2f} seconds.".format(a_length))
        dpg.set_y_scroll("infer_status_window", 1000000)

    def callback_redo_entry(self, sender, app_data, user_data):
        status = dpg.get_value("infer_status_text")
        dpg.set_value("infer_status_text", status + "\nRe-inferring entry with updated text.")
        dpg.set_y_scroll("infer_status_window", 1000000)
        self.stop()
        text = dpg.get_value(user_data["tag_name"])

        #update table array
        td = inferer.get_table_array()
        td[user_data["rank"]][0] = text
        inferer.set_table_array(td)
        inferer.show_table()        

        #run inference again
        result = self.run_inference(text, "text_input", user_data["wav_path"])
        a = AudioSegment.from_file(result[0])    
        t_play = threading.Thread(target=self.play, args=(a,))
        t_play.start()              

    def callback_remove_entry(self, sender, app_data, user_data):
        status = dpg.get_value("infer_status_text")
        dpg.set_value("infer_status_text", status + "\nRemoving entry")
        dpg.set_y_scroll("infer_status_window", 1000000)      
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

    def run_inference(self, input_text, mode, wav_path):
        self.is_running = True
        os.makedirs(dpg.get_value("infer_project_name") + "/wavs_out", exist_ok=True)
        hparams = create_hparams()
        hparams.sampling_rate = 22050
        hparams.p_attention_dropout = 0
        hparams.p_decoder_dropout = 0
        hparams.max_decoder_steps = int(dpg.get_value("infer_max_decoder_steps"))

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

                self.is_running = False
                #copy file
                wav_name =  dpg.get_value("infer_project_name") + "/wavs_out/" + str(self.file_count) + ".wav"
                shutil.move("hifi-gan/generated_files_from_mel/mel.data_generated_e2e.wav", wav_name)
                inferer.file_count += 1
                print("hifi infer done")
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
                print("waveglow infer done")
                self.is_running = False
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

            if p_count == 0:
                #bad file selected!
                status = dpg.get_value("infer_status_text")
                dpg.set_value("infer_status_text", status + "\nError: No splits from text file! Did you select correct file?")
                dpg.set_y_scroll("infer_status_window", 1000000)
                self.is_running = False
                return None

            if phrase_splits:
                result = []
                for i, p in enumerate(phrase_splits):
                    
                    status = dpg.get_value("infer_status_text")
                    dpg.set_value("infer_status_text", status + "\nInferring cut {} of {}.".format(i+1, p_count))
                    dpg.set_y_scroll("infer_status_window", 1000000)

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
                        
                self.is_running = False
                return result


def callback_infer_open_model_taco(sender, app_data):
    d_path = app_data["selections"]
    key = list(d_path.keys())[0]
    path = app_data["selections"][key]
    inferer.set_taco_model_path(path)
    status = dpg.get_value("infer_status_text")
    dpg.set_value("infer_status_text", status + "\nTacotron2 model {} selected.".format(path))
    dpg.set_y_scroll("infer_status_window", 1000000)

def callback_infer_open_model_hifigan(sender, app_data):
    d_path = app_data["selections"]
    key = list(d_path.keys())[0]
    path = app_data["selections"][key]
    inferer.set_hifigan_model_path(path)
    status = dpg.get_value("infer_status_text")
    dpg.set_value("infer_status_text", status + "\nHifi-Gan Model {} selected.".format(path))
    dpg.set_y_scroll("infer_status_window", 1000000)    

def callback_infer_open_model_waveglow(sender, app_data):
    d_path = app_data["selections"]
    key = list(d_path.keys())[0]
    path = app_data["selections"][key]
    inferer.set_waveglow_model_path(path)
    status = dpg.get_value("infer_status_text")
    dpg.set_value("infer_status_text", status + "\nWaveglow Model {} selected.".format(path))
    dpg.set_y_scroll("infer_status_window", 1000000)  

def callback_infer_open_text_file(sender, app_data):
    d_path = app_data["selections"]
    key = list(d_path.keys())[0]
    path = app_data["selections"][key]
    inferer.set_text_file_path(path)
    status = dpg.get_value("infer_status_text")
    dpg.set_value("infer_status_text", status + "\nText file {} selected.".format(path))
    dpg.set_y_scroll("infer_status_window", 1000000)    

def callback_trainer_open_project(sender, app_data):
    path = app_data["file_path_name"]
    # d_path = app_data["selections"]
    # key = list(d_path.keys())[0]
    # path = app_data["selections"][key]
    trainer.set_project_folder(path)
    print(path)
    dpg.set_value("trainer_status_output", dpg.get_value("trainer_status_output") + "\nProject folder {} chosen.".format(path))
    dpg.set_y_scroll("trainer_status_window", 1000000) 

def callback_trainer_open_hifigan_checkpoint(sender, app_data):
    path = app_data["file_path_name"]
    # d_path = app_data["selections"]
    # key = list(d_path.keys())[0]
    # path = app_data["selections"][key]
    trainer.set_hifigan_checkpoint_path(path)
    dpg.set_value("trainer_status_output", dpg.get_value("trainer_status_output") + "\nHifi-Gan checkpoint {} chosen.".format(path))
    dpg.set_y_scroll("trainer_status_window", 1000000)       

def callback_trainer_open_taco_checkpoint(sender, app_data):
    d_path = app_data["selections"]
    key = list(d_path.keys())[0]
    path = app_data["selections"][key]
    trainer.set_taco_checkpoint_path(path)
    dpg.set_value("trainer_status_output", dpg.get_value("trainer_status_output") + "\nTacotron2 checkpoint {} chosen.".format(path))
    dpg.set_y_scroll("trainer_status_window", 1000000) 

def callback_trainer_open_waveglow_checkpoint(sender, app_data):
    d_path = app_data["selections"]
    key = list(d_path.keys())[0]
    path = app_data["selections"][key]
    trainer.set_waveglow_checkpoint_path(path)
    dpg.set_value("trainer_status_output", dpg.get_value("trainer_status_output") + "\nWaveglow checkpoint {} chosen.".format(path))
    dpg.set_y_scroll("trainer_status_window", 1000000) 

def callback_trainer_start_training(sender, data):
    if trainer.is_running():
        return        
    elif not trainer.get_project_folder():
        dpg.set_value("trainer_status_output", dpg.get_value("trainer_status_output") + "\nError: No project folder selected!")
        dpg.set_y_scroll("trainer_status_window", 1000000)         
        return
    elif not trainer.get_hifigan_checkpoint_path() and dpg.get_value("train_models_radio") == "Train Hifi-Gan model":
        dpg.set_value("trainer_status_output", dpg.get_value("trainer_status_output") + "\nTraining new Hifi-Gan model without checkpoint!")
        dpg.set_y_scroll("trainer_status_window", 1000000)       
    elif not trainer.get_taco_checkpoint_path() and dpg.get_value("train_models_radio") == "Train Tacotron2 model":
        dpg.set_value("trainer_status_output", dpg.get_value("trainer_status_output") + "\nTraining new Tacotron2 model without checkpoint!")
        dpg.set_y_scroll("trainer_status_window", 1000000)   
    elif not trainer.get_waveglow_checkpoint_path() and dpg.get_value("train_models_radio") == "Train Waveglow model":
        dpg.set_value("trainer_status_output", dpg.get_value("trainer_status_output") + "\nTraining new Waveglow model without checkpoint!")
        dpg.set_y_scroll("trainer_status_window", 1000000)                           

    batch_size = dpg.get_value("trainer_batch_size")
    iters_per_checkpoint = dpg.get_value("trainer_iters_per_checkpoint")    
    learning_rate = dpg.get_value("trainer_learning_rate_radio")
    multigpu = dpg.get_value("trainer_multigpu")

    trainer.train_model(dpg.get_value("train_models_radio"), batch_size, iters_per_checkpoint, learning_rate, multigpu)

def callback_stop_training(sender, data):
    if trainer.is_running():
        trainer.stop_training()


def callback_trainer_start_tensorboard(sender, data):
    path = trainer.get_project_folder()
    if not path:
        return
    path = path + "/checkpoints/logs"

    print("OPENING TENSORBOARD AT PATH {}".format(path))
    import webbrowser as web
    web.open("http://localhost:6006")
    # out = "tensorboard --logdir '/model utilities/hifi-gan/attenborough.model/logs'"
    # print(out)
    tp = subprocess.Popen(['tensorboard', '--logdir', path])
    trainer.set_tensorboard_process(tp)
    dpg.set_value("trainer_status_output", dpg.get_value("trainer_status_output") + "\nTensorboard process started. Visit http://localhost:6006")
    dpg.set_y_scroll("trainer_status_window", 1000000)     

def callback_trainer_stop_tensorboard(sender, data):
    tp = trainer.get_tensorboard_process()
    tp.kill()    
    dpg.set_value("trainer_status_output", dpg.get_value("trainer_status_output") + "\nTensorboard process stopped.")
    dpg.set_y_scroll("trainer_status_window", 1000000)       

def callback_trainer_pause_training(sender, data):
    trainer.pause_training()


def callback_run_inference(sender, app_data, user_data):

    if inferer.get_is_running():
        return

    # check to see if models selected
    if not inferer.get_hifigan_model_path() and dpg.get_value("infer_model_radio") == "Use Hifi-Gan model":
        status = dpg.get_value("infer_status_text")
        dpg.set_value("infer_status_text", status + "\nError: Hifi-Gan mode but no model chosen!")
        dpg.set_y_scroll("infer_status_window", 1000000)
        return
    if not inferer.get_waveglow_model_path() and dpg.get_value("infer_model_radio") == "Use Waveglow model":
        status = dpg.get_value("infer_status_text")
        dpg.set_value("infer_status_text", status + "\nError: Waveglow mode but no model chosen!")
        dpg.set_y_scroll("infer_status_window", 1000000)            
        return
    if not inferer.get_taco_model_path():
        status = dpg.get_value("infer_status_text")
        dpg.set_value("infer_status_text", status + "\nError: Tacotron2 model not chosen!")
        dpg.set_y_scroll("infer_status_window", 1000000)
        return

    if user_data == "single": 
        t = dpg.get_value("text_input")
        if t:
            print("running inference")            
            status = dpg.get_value("infer_status_text")
            dpg.set_value("infer_status_text", status + "\nRunning inference...")
            dpg.set_y_scroll("infer_status_window", 1000000)

            result = inferer.run_inference(t, "text_input", None)
            a = AudioSegment.from_file(result[0]) 
            t_play = threading.Thread(target=inferer.play, args=(a,))
            t_play.start()
            a_length = a.duration_seconds        
            entry = np.array([t, result[0]])
            inferer.add_entry([entry])
            status = dpg.get_value("infer_status_text")
            dpg.set_value("infer_status_text", status + "\nInferred entry added to table with length {0:.2f} seconds.".format(a_length))
            dpg.set_y_scroll("infer_status_window", 1000000)
        else:
            print("Nothing to infer!")
            status = dpg.get_value("infer_status_text")
            dpg.set_value("infer_status_text", status + "\nError: no input to infer!")
            dpg.set_y_scroll("infer_status_window", 1000000)
            return
    
    elif user_data == "file":
        print(inferer.get_text_file_path())
        if inferer.get_text_file_path():
            if os.path.exists(inferer.get_text_file_path()):
                status = dpg.get_value("infer_status_text")
                dpg.set_value("infer_status_text", status + "\nRunning inference...")
                dpg.set_y_scroll("infer_status_window", 1000000)
                with open(inferer.get_text_file_path(), 'r') as f:
                    r = f.readlines()
                    text_file = " ".join(r)
                    result = inferer.run_inference(text_file, "text_input_file", None)
                    if not result:
                        return
                    for r in result:
                        entry = np.array([r[0], r[1]])
                        inferer.add_entry([entry])
                status = dpg.get_value("infer_status_text")
                dpg.set_value("infer_status_text", status + "\nInferring text file completed.")
                dpg.set_y_scroll("infer_status_window", 1000000)
        else:
            status = dpg.get_value("infer_status_text")
            dpg.set_value("infer_status_text", status + "\nError: Text file not chosen!")
            dpg.set_y_scroll("infer_status_window", 1000000)
            return

def callback_export_infer_table(sender, data):
    ta = inferer.get_table_array()
    if ta.size > 0:
        p_path = dpg.get_value("infer_project_name")
        with open(p_path + "/wav_list.csv", 'w') as f:
            for entry in ta:
                f.write(entry[0])
                f.write('|')
                f.write(entry[1])
                f.write('\n')                
            # t = dpg.get_item_children("infer_table")
            # for i in range(0, len(t[1])):
            #     f.write(dpg.get_value("wav_path_" + str(i)))
            #     f.write('|')
            #     f.write(dpg.get_value("input_text_" + str(i)))
            #     f.write('\n')
        status = dpg.get_value("infer_status_text")
        dpg.set_value("infer_status_text", status + "\nTable exported to project name")
        dpg.set_y_scroll("infer_status_window", 1000000)
    else:
        status = dpg.get_value("infer_status_text")
        dpg.set_value("infer_status_text", status + "\nError: no table to export!")
        dpg.set_y_scroll("infer_status_window", 1000000)

def callback_window_control(sender, data):
    if dpg.is_item_active("inference_tab"):
        dpg.configure_item("infer_status_window", show=True)
        dpg.configure_item("infer_table_window", show=True)
        dpg.configure_item("infer_text_file_window", show=True)
        dpg.configure_item("infer_single_text_window", show=True)
        dpg.configure_item("infer_choose_model_window", show=True)
        dpg.configure_item("trainer_status_window", show=False)
        dpg.configure_item("train_models_project_window", show=False)
        dpg.configure_item("trainer_hparams_window", show=False)
                
        
    elif dpg.is_item_active("train_models_tab"):
        dpg.configure_item("infer_status_window", show=False)
        dpg.configure_item("infer_table_window", show=False)
        dpg.configure_item("infer_text_file_window", show=False)
        dpg.configure_item("infer_single_text_window", show=False)
        dpg.configure_item("infer_choose_model_window", show=False)    
        dpg.configure_item("train_models_project_window", show=True)
        dpg.configure_item("trainer_status_window", show=True)
        dpg.configure_item("trainer_hparams_window", show=True)

with dpg.window(tag='mainwindow', label="Model Utilites", width=1400, height=800):
   
    with dpg.tab_bar(tag="tab_bar_1"):        
            
        with dpg.tab(tag="inference_tab", label=" Run Inference "):

            with dpg.file_dialog(modal=True, height=400, width=800, directory_selector=False, show=False, callback=callback_infer_open_model_taco, tag="infer_open_model_taco"):
                dpg.add_file_extension(".*", color=(255, 255, 255, 255))

            with dpg.file_dialog(modal=True, height=400, width=800, directory_selector=False, show=False, callback=callback_infer_open_model_hifigan, tag="infer_open_model_hifigan"):
                dpg.add_file_extension(".*", color=(255, 255, 255, 255))

            with dpg.file_dialog(modal=True, height=400, width=800, directory_selector=False, show=False, callback=callback_infer_open_model_waveglow, tag="infer_open_model_waveglow"):
                dpg.add_file_extension(".*", color=(255, 255, 255, 255))

            with dpg.file_dialog(modal=True, height=400, width=800, directory_selector=False, show=False, callback=callback_infer_open_text_file, tag="open_text_file"):
                dpg.add_file_extension(".*", color=(255, 255, 255, 255))

            with dpg.file_dialog(modal=True, height=400, width=800, directory_selector=True, show=False, callback=callback_trainer_open_project, tag="trainer_open_project_dialog"):
                dpg.add_file_extension(".*", color=(255, 255, 255, 255))                

            with dpg.file_dialog(modal=True, height=400, width=800, directory_selector=True, show=False, callback=callback_trainer_open_hifigan_checkpoint, tag="trainer_open_hifigan_checkpoint_dialog"):
                dpg.add_file_extension(".*", color=(255, 255, 255, 255))

            with dpg.file_dialog(modal=True, height=400, width=800, directory_selector=False, show=False, callback=callback_trainer_open_taco_checkpoint, tag="trainer_open_taco_checkpoint_dialog"):
                dpg.add_file_extension(".*", color=(255, 255, 255, 255))  
                
            with dpg.file_dialog(modal=True, height=400, width=800, directory_selector=False, show=False, callback=callback_trainer_open_waveglow_checkpoint, tag="trainer_open_waveglow_checkpoint_dialog"):
                dpg.add_file_extension(".*", color=(255, 255, 255, 255))   

            with dpg.window(tag="infer_status_window", show=True, width=800, height=170, pos=(535,35), horizontal_scrollbar=True, menubar=False, no_resize=True, no_title_bar=True, no_move=True, no_scrollbar=False, no_collapse=True, no_close=True):
                dpg.add_text("Status...", tag="infer_status_text")            

            with dpg.window(tag="infer_choose_model_window", show=True, width=525, height=170, pos=(5,35), menubar=False, no_resize=True, no_title_bar=True, no_move=True, no_scrollbar=True, no_collapse=True, no_close=True):     
                
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
                dpg.add_spacer(height=2)
                dpg.add_input_text(width=75, tag="infer_max_decoder_steps", label="Max decoder steps", default_value=10000)

            with dpg.window(tag="infer_single_text_window", show=True, width=850, height=60, pos=(5,210), menubar=False, no_resize=True, no_title_bar=True, no_move=True, no_scrollbar=True, no_collapse=True, no_close=True):     
                dpg.add_text("Input single line text:")             
                dpg.add_input_text(width=800, tag="text_input")
                dpg.add_button(label="Run inference", tag="run_inference_single", callback=callback_run_inference, user_data="single")
            with dpg.window(tag="infer_text_file_window", show=True, width=275, height=100, pos=(860,210), menubar=False, no_resize=True, no_title_bar=True, no_move=True, no_scrollbar=True, no_collapse=True, no_close=True):     
                dpg.add_text("Input text file:")
                with dpg.group(horizontal=True):              
                    dpg.add_button(label="Choose text file", tag="choose_text_file", callback=lambda: dpg.show_item("open_text_file"))
                    dpg.add_button(label="Run inference", tag="run_inference", callback=callback_run_inference, user_data="file")

            with dpg.window(tag="infer_table_window", no_background=True, show=True, width=1350, height=440, pos=(0,310), menubar=False, no_resize=True, no_title_bar=True, no_move=True, no_scrollbar=True, no_collapse=True, no_close=True):     
                dpg.add_button(label="Export .csv", tag="export_infer_table", callback=callback_export_infer_table) 
                with dpg.table(pos=(0, 0), resizable=False, scrollY=True, row_background=True, borders_innerH=True, borders_outerH=True, borders_innerV=True,
                    borders_outerV=True, parent="infer_table_window", header_row=True, width=1325, height=400, tag="infer_table") as infer_table_object:
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=875, parent="infer_table", label='TEXT')
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=250, parent="infer_table", label='AUDIO FILE')
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=200, parent="infer_table", label='OPTIONS')


        with dpg.tab(tag="train_models_tab", label=" Train Models "):  
            with dpg.window(tag="train_models_project_window", show=False, width=600, height=300, pos=(5,35), menubar=False, no_resize=True, no_title_bar=True, no_move=True, no_scrollbar=True, no_collapse=True, no_close=True):     
                dpg.add_text("TRAIN TACOTRON2, HIFI-GAN, OR WAVEGLOW MODELS")
                dpg.add_text("Project folder should contain audio clips in its /wavs directory.\nTraining file should be named 'training.csv' and validation file named 'validation.csv'\nMore information at https://github.com/youmebangbang/deepvoice-model-utilities")
                dpg.add_spacer(height=3)
                dpg.add_button(label="Choose project folder", tag="trainer_open_project", callback=lambda: dpg.show_item("trainer_open_project_dialog"))
                dpg.add_spacer(height=3) 
                with dpg.group(horizontal=True):        
                    dpg.add_radio_button(items=["Train Tacotron2 model", "Train Hifi-Gan model", "Train Waveglow model"], tag="train_models_radio", default_value="Train Tacotron2 model", horizontal=False)
                    with dpg.group():
                        dpg.add_button(label="Choose Tacotron2 checkpoint file", tag="trainer_open_taco_checkpoint", callback=lambda: dpg.show_item("trainer_open_taco_checkpoint_dialog"))
                        dpg.add_button(label="Choose Hifi-Gan checkpoint folder", tag="trainer_open_hifigan_checkpoint", callback=lambda: dpg.show_item("trainer_open_hifigan_checkpoint_dialog"))
                        dpg.add_button(label="Choose Waveglow checkpoint file", tag="trainer_open_waveglow_checkpoint", callback=lambda: dpg.show_item("trainer_open_waveglow_checkpoint_dialog"))
                    
                dpg.add_spacer(height=5)               
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Start training", tag="trainer_start_training", callback=callback_trainer_start_training)
                    dpg.add_button(label="Stop training", tag="trainer_stop_training", callback=callback_stop_training)
                    dpg.add_button(label="Start tensorboard", tag="trainer_start_tensorboard", callback=callback_trainer_start_tensorboard)
                    dpg.add_button(label="Stop tensorboard", tag="trainer_stop_tensorboard", callback=callback_trainer_stop_tensorboard)
                dpg.add_button(label="Pause training", tag="trainer_pause_training", callback=callback_trainer_pause_training)
                
              
            with dpg.window(tag="trainer_status_window", show=False, width=740, height=650, pos=(610,35), horizontal_scrollbar=True, menubar=False, no_resize=True, no_title_bar=True, no_move=True, no_scrollbar=False, no_collapse=True, no_close=True):              
               dpg.add_text("Status...", tag="trainer_status_output")  

            with dpg.window(tag="trainer_hparams_window", show=False, width=600, height=345, pos=(5,340), horizontal_scrollbar=False, menubar=False, no_resize=True, no_title_bar=True, no_move=True, no_scrollbar=True, no_collapse=True, no_close=True):              
                dpg.add_text("TRAINING PARAMETERS")
                dpg.add_spacer(height=3) 
                with dpg.group():
                    dpg.add_input_text(width=50, tag="trainer_batch_size", default_value="184", label="Batch size")
                    dpg.add_spacer(height=3) 
                    dpg.add_input_text(width=100, tag="trainer_iters_per_checkpoint", default_value="500", label="Iterations per checkpoint")
                    dpg.add_spacer(height=3) 
                    with dpg.group(horizontal=True):
                        dpg.add_text("Choose learning rate:")
                        dpg.add_radio_button(items=["2e-4", "1e-4", "5e-5", "1e-5", "5e-6", "1e-6"], tag="trainer_learning_rate_radio", default_value="1e-4", horizontal=False)
                    dpg.add_spacer(height=3) 
                    dpg.add_checkbox(tag="trainer_multigpu", default_value=False, label="Multi-GPU (linux only)")
                    dpg.add_checkbox(tag="trainer_warmstart", default_value=False, label="Warmstart training (no speaker embeddings)")


                
inferer = Inferer()
trainer = Trainer()   

with dpg.item_handler_registry(tag="window_handler"):
    dpg.add_item_active_handler(callback=callback_window_control)
dpg.bind_item_handler_registry("train_models_tab", "window_handler")
dpg.bind_item_handler_registry("inference_tab", "window_handler")

with dpg.font_registry():
    default_font = dpg.add_font("CheyenneSans-Light.otf", 17)
    font2 = dpg.add_font("PublicSans-Regular.otf", 18)
    #font3 = dpg.add_font("VarelaRound-Regular.ttf", 17)
    

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

def exit_dpg(sender, data):
    if trainer.is_running():
        trainer.stop_training()

dpg.set_exit_callback(exit_dpg)

dpg.create_viewport(title="Deep Voice Model Utilities v1.0 by YouMeBangBang", width=1400, height=800)
dpg.setup_dearpygui()
dpg.show_viewport()

dpg.set_global_font_scale(1.0)

dpg.set_primary_window("mainwindow", True)
dpg.start_dearpygui()
dpg.destroy_context()


