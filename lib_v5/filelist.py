import json

def get_vr_download_list(list):
    with open("lib_v5/filelists/download_lists/vr_download_list.txt", "r") as f:
        text=f.read().splitlines()

    list = text
    
    return list

def get_mdx_download_list(list):
    with open("lib_v5/filelists/download_lists/mdx_download_list.txt", "r") as f:
        text=f.read().splitlines()

    list = text
    
    return list

def get_demucs_download_list(list):
    with open("lib_v5/filelists/download_lists/demucs_download_list.txt", "r") as f:
        text=f.read().splitlines()

    list = text
    
    return list

def get_mdx_demucs_en_list(list):
    with open("lib_v5/filelists/ensemble_list/mdx_demuc_en_list.txt", "r") as f:
        text=f.read().splitlines()

    list = text
    
    return list

def get_vr_en_list(list):
    with open("lib_v5/filelists/ensemble_list/vr_en_list.txt", "r") as f:
        text=f.read().splitlines()

    list = text
    
    return list

def get_download_links(links, downloads=''):

    f = open(f"lib_v5/filelists/download_lists/download_links.json")
    download_links = json.load(f)

    if downloads == 'Demucs v3: mdx':
        url_1 = download_links['Demucs_v3_mdx_url_1']
        url_2 = download_links['Demucs_v3_mdx_url_2']
        url_3 = download_links['Demucs_v3_mdx_url_3']
        url_4 = download_links['Demucs_v3_mdx_url_4']
        url_5 = download_links['Demucs_v3_mdx_url_5']
        
        links = url_1, url_2, url_3, url_4, url_5
        
        
    if downloads == 'Demucs v3: mdx_q':
        url_1 = download_links['Demucs_v3_mdx_q_url_1']
        url_2 = download_links['Demucs_v3_mdx_q_url_2']
        url_3 = download_links['Demucs_v3_mdx_q_url_3']
        url_4 = download_links['Demucs_v3_mdx_q_url_4']
        url_5 = download_links['Demucs_v3_mdx_q_url_5']
        
        links = url_1, url_2, url_3, url_4, url_5
        
    if downloads == 'Demucs v3: mdx_extra':
        url_1 = download_links['Demucs_v3_mdx_extra_url_1']
        url_2 = download_links['Demucs_v3_mdx_extra_url_2']
        url_3 = download_links['Demucs_v3_mdx_extra_url_3']
        url_4 = download_links['Demucs_v3_mdx_extra_url_4']
        url_5 = download_links['Demucs_v3_mdx_extra_url_5']
        
        links = url_1, url_2, url_3, url_4, url_5
        
    if downloads == 'Demucs v3: mdx_extra_q':
        url_1 = download_links['Demucs_v3_mdx_extra_q_url_1']
        url_2 = download_links['Demucs_v3_mdx_extra_q_url_2']
        url_3 = download_links['Demucs_v3_mdx_extra_q_url_3']
        url_4 = download_links['Demucs_v3_mdx_extra_q_url_4']
        url_5 = download_links['Demucs_v3_mdx_extra_q_url_5']
        
        links = url_1, url_2, url_3, url_4, url_5
        
    if downloads == 'Demucs v3: UVR Models':
        url_1 = download_links['Demucs_v3_UVR_url_1']
        url_2 = download_links['Demucs_v3_UVR_url_2']
        url_3 = download_links['Demucs_v3_UVR_url_3']
        url_4 = download_links['Demucs_v3_UVR_url_4']
        url_5 = download_links['Demucs_v3_UVR_url_5']

        links = url_1, url_2, url_3, url_4, url_5
    
    if downloads == 'Demucs v2: demucs':
        url_1 = download_links['Demucs_v2_demucs_url_1']
        links = url_1
        
    if downloads == 'Demucs v2: demucs_extra':
        url_1 = download_links['Demucs_v2_demucs_extra_url_1']

        links = url_1
        
    if downloads == 'Demucs v2: demucs48_hq':
        url_1 = download_links['Demucs_v2_demucs48_hq_url_1']

        links = url_1
        
    if downloads == 'Demucs v2: tasnet':
        url_1 = download_links['Demucs_v2_tasnet_url_1']

        links = url_1
        
    if downloads == 'Demucs v2: tasnet_extra':
        url_1 = download_links['Demucs_v2_tasnet_extra_url_1']

        links = url_1
        
    if downloads == 'Demucs v2: demucs_unittest':
        url_1 = download_links['Demucs_v2_demucs_unittest_url_1']

        links = url_1
        
    if downloads == 'Demucs v1: demucs':
        url_1 = download_links['Demucs_v1_demucs_url_1']

        links = url_1
        
    if downloads == 'Demucs v1: demucs_extra':
        url_1 = download_links['Demucs_v1_demucs_extra_url_1']

        links = url_1
        
    if downloads == 'Demucs v1: light':
        url_1 = download_links['Demucs_v1_light_url_1']

        links = url_1
        
    if downloads == 'Demucs v1: light_extra':
        url_1 = download_links['Demucs_v1_light_extra_url_1']

        links = url_1
        
    if downloads == 'Demucs v1: tasnet':
        url_1 = download_links['Demucs_v1_tasnet_url_1']

        links = url_1
        
    if downloads == 'Demucs v1: tasnet_extra':
        url_1 = download_links['Demucs_v1_tasnet_extra_url_1']
        
        links = url_1
        
    if downloads == 'model_repo':
        url_1 = download_links['model_repo_url_1']
        
        links = url_1
        
    if downloads == 'single_model_repo':
        url_1 = download_links['single_model_repo_url_1']
        
        links = url_1
        
    if downloads == 'exclusive':
        url_1 = download_links['exclusive_url_1']
        url_2 = download_links['exclusive_url_2']
        
        links = url_1, url_2, url_3
        
    if downloads == 'refresh':
        url_1 = download_links['refresh_url_1']
        url_2 = download_links['refresh_url_2']
        url_3 = download_links['refresh_url_3']
        
        links = url_1, url_2, url_3

    if downloads == 'app_patch':
        url_1 = download_links['app_patch']
        
        links = url_1
        
    return links

def provide_model_param_hash(model_hash):                          
        #v5 Models
        if model_hash == '47939caf0cfe52a0e81442b85b971dfd':  
            model_params_set=str('lib_v5/modelparams/4band_44100.json')
            param_name=str('4band_44100')
        elif model_hash == '4e4ecb9764c50a8c414fee6e10395bbe':  
            model_params_set=str('lib_v5/modelparams/4band_v2.json')
            param_name=str('4band_v2')
        elif model_hash == 'e60a1e84803ce4efc0a6551206cc4b71':  
            model_params_set=str('lib_v5/modelparams/4band_44100.json')
            param_name=str('4band_44100')
        elif model_hash == 'a82f14e75892e55e994376edbf0c8435':  
            model_params_set=str('lib_v5/modelparams/4band_44100.json')
            param_name=str('4band_44100')
        elif model_hash == '6dd9eaa6f0420af9f1d403aaafa4cc06':   
            model_params_set=str('lib_v5/modelparams/4band_v2_sn.json')
            param_name=str('4band_v2_sn')
        elif model_hash == '5c7bbca45a187e81abbbd351606164e5':    
            model_params_set=str('lib_v5/modelparams/3band_44100_msb2.json')
            param_name=str('3band_44100_msb2')
        elif model_hash == 'd6b2cb685a058a091e5e7098192d3233':    
            model_params_set=str('lib_v5/modelparams/3band_44100_msb2.json')
            param_name=str('3band_44100_msb2')
        elif model_hash == 'c1b9f38170a7c90e96f027992eb7c62b': 
            model_params_set=str('lib_v5/modelparams/4band_44100.json')
            param_name=str('4band_44100')
        elif model_hash == 'c3448ec923fa0edf3d03a19e633faa53':  
            model_params_set=str('lib_v5/modelparams/4band_44100.json')
            param_name=str('4band_44100')
        elif model_hash == '68aa2c8093d0080704b200d140f59e54':  
            model_params_set=str('lib_v5/modelparams/3band_44100.json')
            param_name=str('3band_44100.json')
        elif model_hash == 'fdc83be5b798e4bd29fe00fe6600e147':  
            model_params_set=str('lib_v5/modelparams/3band_44100_mid.json')
            param_name=str('3band_44100_mid.json')
        elif model_hash == '2ce34bc92fd57f55db16b7a4def3d745':  
            model_params_set=str('lib_v5/modelparams/3band_44100_mid.json')
            param_name=str('3band_44100_mid.json')
        elif model_hash == '52fdca89576f06cf4340b74a4730ee5f':  
            model_params_set=str('lib_v5/modelparams/4band_44100.json')
            param_name=str('4band_44100.json')
        elif model_hash == '41191165b05d38fc77f072fa9e8e8a30':  
            model_params_set=str('lib_v5/modelparams/4band_44100.json')
            param_name=str('4band_44100.json')
        elif model_hash == '89e83b511ad474592689e562d5b1f80e':  
            model_params_set=str('lib_v5/modelparams/2band_32000.json')
            param_name=str('2band_32000.json')
        elif model_hash == '0b954da81d453b716b114d6d7c95177f':  
            model_params_set=str('lib_v5/modelparams/2band_32000.json')
            param_name=str('2band_32000.json')
            
        #v4 Models
            
        elif model_hash == '6a00461c51c2920fd68937d4609ed6c8':  
            model_params_set=str('lib_v5/modelparams/1band_sr16000_hl512.json')
            param_name=str('1band_sr16000_hl512')
        elif model_hash == '0ab504864d20f1bd378fe9c81ef37140':  
            model_params_set=str('lib_v5/modelparams/1band_sr32000_hl512.json')
            param_name=str('1band_sr32000_hl512')
        elif model_hash == '7dd21065bf91c10f7fccb57d7d83b07f':  
            model_params_set=str('lib_v5/modelparams/1band_sr32000_hl512.json')
            param_name=str('1band_sr32000_hl512')
        elif model_hash == '80ab74d65e515caa3622728d2de07d23':  
            model_params_set=str('lib_v5/modelparams/1band_sr32000_hl512.json')
            param_name=str('1band_sr32000_hl512')
        elif model_hash == 'edc115e7fc523245062200c00caa847f':  
            model_params_set=str('lib_v5/modelparams/1band_sr33075_hl384.json')
            param_name=str('1band_sr33075_hl384')
        elif model_hash == '28063e9f6ab5b341c5f6d3c67f2045b7':  
            model_params_set=str('lib_v5/modelparams/1band_sr33075_hl384.json')
            param_name=str('1band_sr33075_hl384')
        elif model_hash == 'b58090534c52cbc3e9b5104bad666ef2':  
            model_params_set=str('lib_v5/modelparams/1band_sr44100_hl512.json')
            param_name=str('1band_sr44100_hl512')
        elif model_hash == '0cdab9947f1b0928705f518f3c78ea8f':  
            model_params_set=str('lib_v5/modelparams/1band_sr44100_hl512.json')
            param_name=str('1band_sr44100_hl512')
        elif model_hash == 'ae702fed0238afb5346db8356fe25f13':  
            model_params_set=str('lib_v5/modelparams/1band_sr44100_hl1024.json')
            param_name=str('1band_sr44100_hl1024')       
        else:
            try:
                with open(f"lib_v5/filelists/model_cache/vr_param_cache/{model_hash}.txt", "r") as f:
                    name = f.read()
                model_params_set=str(f'lib_v5/modelparams/{name}')
                param_name=str(name)
                ('using text of hash worked')
            except:            
                model_params_set=str('Not Found Using Hash')
                param_name=str('Not Found Using Hash')
            
        model_params = model_params_set, param_name
                  
        return model_params
            
def provide_model_param_name(ModelName): 
        #1 Band
        if '1band_sr16000_hl512' in ModelName:  
            model_params_set=str('lib_v5/modelparams/1band_sr16000_hl512.json')
            param_name=str('1band_sr16000_hl512')
        elif '1band_sr32000_hl512' in ModelName:  
            model_params_set=str('lib_v5/modelparams/1band_sr32000_hl512.json')
            param_name=str('1band_sr32000_hl512')
        elif '1band_sr33075_hl384' in ModelName:  
            model_params_set=str('lib_v5/modelparams/1band_sr33075_hl384.json')
            param_name=str('1band_sr33075_hl384')
        elif '1band_sr44100_hl256' in ModelName:  
            model_params_set=str('lib_v5/modelparams/1band_sr44100_hl256.json')
            param_name=str('1band_sr44100_hl256')
        elif '1band_sr44100_hl512' in ModelName:  
            model_params_set=str('lib_v5/modelparams/1band_sr44100_hl512.json')
            param_name=str('1band_sr44100_hl512')
        elif '1band_sr44100_hl1024' in ModelName:  
            model_params_set=str('lib_v5/modelparams/1band_sr44100_hl1024.json')
            param_name=str('1band_sr44100_hl1024')
            
        #2 Band
        elif '2band_44100_lofi' in ModelName:  
            model_params_set=str('lib_v5/modelparams/2band_44100_lofi.json')
            param_name=str('2band_44100_lofi')
            
        #3 Band   

        elif '3band_44100_mid' in ModelName:  
            model_params_set=str('lib_v5/modelparams/3band_44100_mid.json')
            param_name=str('3band_44100_mid')
        elif '3band_44100_msb2' in ModelName:  
            model_params_set=str('lib_v5/modelparams/3band_44100_msb2.json')
            param_name=str('3band_44100_msb2')
            
        #4 Band    

        elif '4band_44100_msb' in ModelName:  
            model_params_set=str('lib_v5/modelparams/4band_44100_msb.json')
            param_name=str('4band_44100_msb')
        elif '4band_44100_msb2' in ModelName:  
            model_params_set=str('lib_v5/modelparams/4band_44100_msb2.json')
            param_name=str('4band_44100_msb2')
        elif '4band_44100_reverse' in ModelName:  
            model_params_set=str('lib_v5/modelparams/4band_44100_reverse.json')
            param_name=str('4band_44100_reverse')
        elif 'tmodelparam' in ModelName:  
            model_params_set=str('lib_v5/modelparams/tmodelparam.json')
            param_name=str('User Model Param Set')
        else:
            model_params_set=str('Not Found Using Name')
            param_name=str('Not Found Using Name')
                  
        model_params = model_params_set, param_name
                  
        return model_params
    
def provide_mdx_model_param_name(modelhash):
    with open("lib_v5/filelists/hashes/mdx_original_hashes.txt", "r") as f:
        mdx_original=f.read() 
    with open("lib_v5/filelists/hashes/mdx_new_hashes.txt", "r") as f:
        mdx_new=f.read()
    with open("lib_v5/filelists/hashes/mdx_new_inst_hashes.txt", "r") as f:
        mdx_new_inst=f.read()
        
    if modelhash in mdx_original:
        MDX_modeltype = 'mdx_original'
    elif modelhash in mdx_new:
        MDX_modeltype = 'mdx_new' 
    elif modelhash in mdx_new_inst:
        MDX_modeltype = 'mdx_new_inst' 
    else:
        MDX_modeltype = 'None'
    
    if MDX_modeltype == 'mdx_original':
        modeltype = 'v'
        noise_pro = 'MDX-NET_Noise_Profile_14_kHz'
        stemset_n = '(Vocals)'
        compensate = 1.03597672895
        source_val = 3
        n_fft_scale_set=6144 
        dim_f_set=2048
    elif MDX_modeltype == 'mdx_new':
        modeltype = 'v'
        noise_pro = 'MDX-NET_Noise_Profile_17_kHz'
        stemset_n = '(Vocals)'
        compensate = 1.08
        source_val = 3
        n_fft_scale_set=7680 
        dim_f_set=3072
    elif MDX_modeltype == 'mdx_new_inst':
        modeltype = 'v'
        noise_pro = 'MDX-NET_Noise_Profile_17_kHz'
        stemset_n = '(Instrumental)'
        compensate = 1.08
        source_val = 3
        n_fft_scale_set=7680 
        dim_f_set=3072
    elif modelhash == '6f7eefc2e6b9d819ba88dc0578056ca5':
        modeltype = 'o'
        noise_pro = 'MDX-NET_Noise_Profile_Full_Band'
        stemset_n = '(Other)'
        compensate = 1.03597672895
        source_val = 2
        n_fft_scale_set=8192 
        dim_f_set=2048
    elif modelhash == '72a27258a69b2381b60523a50982e9f1':
        modeltype = 'd'
        noise_pro = 'MDX-NET_Noise_Profile_Full_Band'
        stemset_n = '(Drums)'
        compensate = 1.03597672895
        source_val = 1
        n_fft_scale_set=4096 
        dim_f_set=2048
    elif modelhash == '7051d7315c04285e94a97edcac3f2f76':
        modeltype = 'b'
        noise_pro = 'MDX-NET_Noise_Profile_Full_Band'
        stemset_n = '(Bass)'
        compensate = 1.03597672895
        source_val = 0
        n_fft_scale_set=16384 
        dim_f_set=2048
    else:
        try:
            f = open(f"lib_v5/filelists/model_cache/mdx_model_cache/{modelhash}.json")
            mdx_model_de = json.load(f)
            modeltype = mdx_model_de["modeltype"]
            noise_pro = mdx_model_de["noise_pro"]
            stemset_n = mdx_model_de["stemset_n"]
            compensate = mdx_model_de["compensate"]
            source_val = mdx_model_de["source_val"]
            n_fft_scale_set = mdx_model_de["n_fft_scale_set"]
            dim_f_set = mdx_model_de["dim_f_set"]   
        except:
            modeltype = 'Not Set'
            noise_pro = 'Not Set'
            stemset_n = 'Not Set'
            compensate = 'Not Set'
            source_val = 'Not Set'
            n_fft_scale_set='Not Set'
            dim_f_set='Not Set'
        
        
    model_params = modeltype, noise_pro, stemset_n, compensate, source_val, n_fft_scale_set, dim_f_set
                
    return model_params