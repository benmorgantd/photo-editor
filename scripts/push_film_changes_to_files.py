import os
import json

folder = r'C:\Users\bmorgan\AppData\Local\Temp\photo_editor_session_cache\preset_manifests'

film_sim_data = {
    # 1. Tonality & Color Science
            "rgb_curves" : {
                "r" : [[0.0, 0.25, 0.50, 0.75, 1.0], [0.0, 0.25, 0.50, 0.75, 1.0]],
                "g" : [[0.0, 0.25, 0.50, 0.75, 1.0], [0.0, 0.25, 0.50, 0.75, 1.0]],
                "b" : [[0.0, 0.25, 0.50, 0.75, 1.0], [0.0, 0.25, 0.50, 0.75, 1.0]]
            },
            "color_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0,  0.0],
                [0.0, 0.0,  1.0]
            ],
            
            # 2. Optical Bloom
            "enable_bloom": True,
            "bloom_threshold": 0.70,
            "bloom_radius": 15.0,
            "bloom_strength": 0.20, 
            
            # 3. Selective Halation
            "enable_halation": True,
            "halation_threshold": 0.60,
            "halation_radius": 12.0,
            "halation_strength": 0.35,
            "halation_offset_x": 1.5,
            "halation_offset_y": 0.2,
            
            # 4. Smart Grain
            "enable_grain": True,
            "grain_strength": 0.05, 
            "grain_size": 0.08,      # >1.0 scales grain up for coarse emulsion
            "grain_chroma": 0.15,   # 0.0 = B&W grain, 1.0 = color grain
            
            # 5. Vignette & Optical Softness
            "enable_vignette": True,
            "vignette_strength": 0.35,
            "vignette_radius": 0.75,
            "vignette_softness": 0.45,
            "corner_blur_radius": 1.0
        }

for filename in os.listdir(folder):
    print(filename)
    path = os.path.join(folder, filename)

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for key, value in film_sim_data.items():
        data[key] = value
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
