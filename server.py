# import http.server
# import socketserver

# PORT = 8000

# Handler = http.server.SimpleHTTPRequestHandler

# with socketserver.TCPServer(("", PORT), Handler) as httpd:
#     print("Server started at localhost:" + str(PORT))
#     httpd.serve_forever()


# #test

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from PIL import Image # PIL היא ספרייה לעבודה עם תמונות
import io

# --- 1. הגדרת ארכיטקטורת המודל ---
# חייבים להגדיר את אותה ארכיטקטורה בדיוק כמו באימון
# כדי ש-PyTorch יידע איך לטעון את המשקולות
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc1(x)
        return x

# רשימת הקטגוריות שלנו (לפי הסדר)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# --- 2. טעינת המודל המאומן ---
MODEL_PATH = "cifar10_model.pth"
model = Net() # יצירת מופע "ריק" של הארכיטקטורה
model.load_state_dict(torch.load(MODEL_PATH)) # טעינת המשקולות השמורות
model.eval() # חשוב: העברת המודל למצב "הערכה" (לא אימון)

print("--- המודל נטען, השרת מוכן ---")

# --- 3. הגדרת טרנספורמציות לתמונה ---
# אלו אותן טרנספורמציות שעשינו באימון
transform = transforms.Compose(
    [transforms.Resize((32, 32)), # נוודא שהתמונה בגודל 32x32
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def process_image(image_bytes):
    # פתיחת התמונה מהמידע הבינארי שקיבלנו
    image = Image.open(io.BytesIO(image_bytes))
    # המרה ל-RGB (למקרה שזה PNG עם ערוץ שקיפות)
    image = image.convert('RGB')
    # החלת הטרנספורמציות
    image_tensor = transform(image)
    # הוספת "ממד" של Batch (המודל מצפה לקבל "מנה" של תמונות)
    return image_tensor.unsqueeze(0)

# --- 4. אתחול השרת (Flask) ---
app = Flask(__name__)

# --- 5. הגדרת נקודת הקצה (API Route) ---
@app.route("/predict", methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'}), 400
    
    file = request.files['image']
    image_bytes = file.read()
    
    try:
        # 1. עיבוד התמונה
        image_tensor = process_image(image_bytes)
        
        # 2. הרצת המודל (בלי לחשב נגזרות)
        with torch.no_grad():
            outputs = model(image_tensor)
            
            # 3. קבלת הניחוש (האינדקס עם הציון הגבוה ביותר)
            _, predicted_index = torch.max(outputs.data, 1)
            
            # 4. תרגום האינדקס לשם הקטגוריה
            predicted_class = classes[predicted_index.item()]
            
            # 5. החזרת תשובת JSON
            return jsonify({
                'prediction': predicted_class,
                'class_index': predicted_index.item()
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 6. הרצת השרת ---
if __name__ == '__main__':
    # '0.0.0.0' גורם לשרת להאזין בכל הכתובות, לא רק localhost
    app.run(host='0.0.0.0', port=5000, debug=True)