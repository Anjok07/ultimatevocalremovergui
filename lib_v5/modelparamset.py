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
        elif '2band_32000' in ModelName:  
            model_params_set=str('lib_v5/modelparams/2band_32000.json')
            param_name=str('2band_32000')
        elif '2band_48000' in ModelName:  
            model_params_set=str('lib_v5/modelparams/2band_48000.json')
            param_name=str('2band_48000')
            
        #3 Band   
        elif '3band_44100' in ModelName:  
            model_params_set=str('lib_v5/modelparams/3band_44100.json')
            param_name=str('3band_44100')
        elif '3band_44100_mid' in ModelName:  
            model_params_set=str('lib_v5/modelparams/3band_44100_mid.json')
            param_name=str('3band_44100_mid')
        elif '3band_44100_msb2' in ModelName:  
            model_params_set=str('lib_v5/modelparams/3band_44100_msb2.json')
            param_name=str('3band_44100_msb2')
            
        #4 Band    
        elif '4band_44100' in ModelName:  
            model_params_set=str('lib_v5/modelparams/4band_44100.json')
            param_name=str('4band_44100')
        elif '4band_44100_mid' in ModelName:  
            model_params_set=str('lib_v5/modelparams/4band_44100_mid.json')
            param_name=str('4band_44100_mid')
        elif '4band_44100_msb' in ModelName:  
            model_params_set=str('lib_v5/modelparams/4band_44100_msb.json')
            param_name=str('4band_44100_msb')
        elif '4band_44100_msb2' in ModelName:  
            model_params_set=str('lib_v5/modelparams/4band_44100_msb2.json')
            param_name=str('4band_44100_msb2')
        elif '4band_44100_reverse' in ModelName:  
            model_params_set=str('lib_v5/modelparams/4band_44100_reverse.json')
            param_name=str('4band_44100_reverse')
        elif '4band_44100_sw' in ModelName:  
            model_params_set=str('lib_v5/modelparams/4band_44100_sw.json') 
            param_name=str('4band_44100_sw')
        elif '4band_v2' in ModelName:  
            model_params_set=str('lib_v5/modelparams/4band_v2.json')
            param_name=str('4band_v2')
        elif '4band_v2_sn' in ModelName:  
            model_params_set=str('lib_v5/modelparams/4band_v2_sn.json')
            param_name=str('4band_v2_sn')
        elif 'tmodelparam' in ModelName:  
            model_params_set=str('lib_v5/modelparams/tmodelparam.json')
            param_name=str('User Model Param Set')
        else:
            model_params_set=str('Not Found Using Name')
            param_name=str('Not Found Using Name')
                  
        model_params = model_params_set, param_name
                  
        return model_params