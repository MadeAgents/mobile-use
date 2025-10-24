from PIL import Image
import numpy as np
import re, json
def compare_image(img1: Image.Image, img2: Image.Image):
    img1 = img1.convert('L')
    img2 = img2.convert('L')
    img1 = np.array(img1)
    img2 = np.array(img2)
    return np.array_equal(img1, img2)

def parse_response(content: str, size: tuple[float, float], raw_size: tuple[float, float]):
    action_s = re.findall(r'```JSON(.*)```', content, re.DOTALL)[0].strip()
    action = json.loads(action_s)
    name = action['arguments']['action']
    
    action['arguments'].pop('action')
    params = action['arguments']

    for k, v in params.items():
        if k in ['coordinate', 'coordinate2', 'point', 'start_point', 'end_point']:
            try:
                x = round(v[0] / size[0] * raw_size[0])
                y = round(v[1] / size[1] * raw_size[1])
                params[k] = (x, y)
            except:
                pass

    return name, params