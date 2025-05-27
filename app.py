
from flask import Flask, render_template, request
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def get_point(landmarks, index, w, h):
    lm = landmarks.landmark[index]
    return int(lm.x * w), int(lm.y * h)

def draw_text_on_image(img_path, text, output_path='static/with_text.jpg'):
    font_path = "./static/NanumGothic.ttf"
    font = ImageFont.truetype(font_path, 20)
    cv_img = cv2.imread(img_path)
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    lines = text.split('\n')
    x, y = 20, 20
    for line in lines:
        draw.text((x, y), line, font=font, fill=(255, 0, 0))
        y += 30

    result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_img)
    return output_path

def refine_with_gpt(messages: list):
    prompt = (
        "당신은 관상 전문가입니다. 아래 얼굴 특징을 바탕으로 자연스럽고 신뢰감 있는 말투로 관상 분석을 해주세요.\n"
        "문장은 3~4개로 구성하며, 각각의 특징이 잘 드러나도록 해주세요.\n\n"
    )

    for line in messages:
        prompt += f"- {line}\n"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    filepath = 'static/result.jpg'
    file.save(filepath)

    image = cv2.imread(filepath)
    image = cv2.resize(image, (640, 480))
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    results_text, extended_results = [], []

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            jaw_left = get_point(landmarks, 234, w, h)
            jaw_right = get_point(landmarks, 454, w, h)
            chin = get_point(landmarks, 152, w, h)
            forehead = get_point(landmarks, 10, w, h)
            nose_tip = get_point(landmarks, 1, w, h)
            nose_top = get_point(landmarks, 6, w, h)
            eye_left = get_point(landmarks, 33, w, h)
            eye_right = get_point(landmarks, 263, w, h)
            eye_inner_left = get_point(landmarks, 133, w, h)
            mouth_left = get_point(landmarks, 61, w, h)
            mouth_right = get_point(landmarks, 291, w, h)

            def euclidean(p1, p2):
                return int(((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5)

            jaw_width = euclidean(jaw_left, jaw_right)
            face_height = euclidean(forehead, chin)
            eye_gap = abs(eye_left[0] - eye_right[0])
            nose_length = abs(nose_top[1] - nose_tip[1])
            mouth_slope = mouth_right[1] - mouth_left[1]
            eye_slope = eye_inner_left[1] - eye_left[1]

            if jaw_width > 270:
                results_text.append("💪 턱이 넓은 편이라 리더십과 추진력이 강합니다.")
            elif jaw_width < 220:
                results_text.append("😊 턱이 갸름해 감수성이 풍부하고 섬세한 성향입니다.")
            else:
                results_text.append("🙂 균형 잡힌 턱선으로 조화로운 성격입니다.")

            if face_height > 330:
                results_text.append("🧠 얼굴이 긴 편으로 사고 중심의 이성적인 스타일입니다.")
            else:
                results_text.append("😄 얼굴이 짧은 편으로 행동력과 친근함이 돋보입니다.")

            if eye_gap > 150:
                results_text.append("👀 눈 사이가 넓어 독립적이고 분석적인 성격입니다.")
            else:
                results_text.append("👀 눈 사이가 가까워 감성적이고 사람 중심적인 성향입니다.")

            if nose_length > 40:
                results_text.append("👃 콧대가 높고 길어 자존감과 자기 통제력이 뛰어납니다.")
            else:
                results_text.append("👃 콧대가 짧은 편이라 유연하고 포용력이 강한 스타일입니다.")

            if eye_slope < -5:
                extended_results.append("👁 눈꼬리가 올라가 활기차고 외향적인 성격입니다.")
            elif eye_slope > 5:
                extended_results.append("👁 눈꼬리가 내려가 차분하고 온화한 성향입니다.")
            else:
                extended_results.append("👁 눈꼬리가 수평이라 침착하고 균형 잡힌 스타일입니다.")

            if mouth_slope < -5:
                extended_results.append("😊 입꼬리가 올라가 밝고 긍정적인 사람입니다.")
            elif mouth_slope > 5:
                extended_results.append("😐 입꼬리가 내려가 조용하고 신중한 스타일입니다.")
            else:
                extended_results.append("🙂 입꼬리가 평평해 차분하고 믿음직한 인상을 줍니다.")

    refined_text = refine_with_gpt(results_text + extended_results)
    draw_text_on_image(filepath, refined_text)
    return render_template('index.html', filename='with_text.jpg', final_analysis=refined_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
