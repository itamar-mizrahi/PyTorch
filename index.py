# import torch
# print("Hello from the Docker container!")
# x = torch.rand(5, 3)
# print(x)

# import torch

# # 1. ניצור טנזור, ונגיד ל-PyTorch "לעקוב" אחריו
# #    requires_grad=True זה כמו להפעיל "הקלטה"
# x = torch.tensor(2.0, requires_grad=True)

# # 2. נגדיר פונקציה מתמטית פשוטה: y = x^2 + 3
# y = x**2 + 3

# # 3. כאן הקסם: אנו מורים ל-PyTorch לחשב את הנגזרת
# #    של y ביחס ל-x בנקודה שבה x=2.
# y.backward()

# # 4. נדפיס את הנגזרת (השיפוע) ש-PyTorch חישב עבור x
# print(x.grad)


# import torch

# # --- 1. הכנת הנתונים ---
# # אלו הנתונים ה"אמיתיים" שלנו. אנחנו רוצים למצוא קו שעובר דרכם.
# # נניח ש-y = 2*x + 1 (במציאות לא נדע את זה)
# X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
# Y = torch.tensor([[3.0], [5.0], [7.0], [9.0]], dtype=torch.float32)

# # --- 2. אתחול המודל (הניחושים ההתחלתיים) ---
# # אנחנו רוצים למצוא את m ו-b. ב-PyTorch קוראים להם "משקולות" (weights) ו-"הטיה" (bias).
# # נתחיל עם ניחושים אקראיים ונגיד ל-PyTorch לעקוב אחריהם.
# W = torch.randn(1, 1, requires_grad=True, dtype=torch.float32) # 'm' שלנו
# b = torch.randn(1, 1, requires_grad=True, dtype=torch.float32) # 'b' שלנו

# print(f"ניחוש התחלתי: W={W.item():.3f}, b={b.item():.3f}")

# # --- 3. הגדרת כלי העבודה ---
# learning_rate = 0.1  # כמה מהר נתקן את הטעויות (גודל הצעד)
# n_epochs = 100        # כמה פעמים נחזור על כל הנתונים

# # --- 4. לולאת האימון (כאן קורה הקסם) ---
# for epoch in range(n_epochs):
#     # א. חישוב הניחוש (y_pred) על בסיס ה-W ו-b הנוכחיים
#     # y_pred = W * X + b
#     Y_pred = torch.matmul(X, W) + b

#     # ב. חישוב ה"טעות" (loss) - כמה רחוק הניחוש מהאמת
#     # (Y_pred - Y)^2
#     loss = torch.mean((Y_pred - Y)**2)

#     # ג. חישוב הנגזרות (איך כל פרמטר השפיע על הטעות)
#     # זה ה-autograd שראינו קודם!
#     loss.backward()

#     # ד. עדכון הניחושים (W ו-b) נגד כיוון הנגזרת
#     # אנחנו עושים את זה בתוך 'no_grad' כי זה לא חלק מה"הקלטה"
#     with torch.no_grad():
#         W -= learning_rate * W.grad
#         b -= learning_rate * b.grad

#     # ה. איפוס הנגזרות לפעם הבאה
#     W.grad.zero_()
#     b.grad.zero_()

#     # הדפסת התקדמות
#     if (epoch + 1) % 10 == 0:
#         print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")

# # --- 5. תוצאה סופית ---
# print("\n--- אימון הסתיים ---")
# print(f"תוצאה אמיתית:  W=2.0, b=1.0")
# print(f"תוצאה שנלמדה: W={W.item():.3f}, b={b.item():.3f}")

# import torch
# import torch.nn as nn  # ייבוא הרכיבים המובנים

# # --- 1. הכנת הנתונים (ללא שינוי) ---
# X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
# Y = torch.tensor([[3.0], [5.0], [7.0], [9.0]], dtype=torch.float32)

# # --- 2. הגדרת המודל (הדרך החדשה) ---
# # W ו-b מוסתרים עכשיו בתוך 'nn.Linear'
# # in_features=1 : מצפה ל-X עם תכונה אחת (מספר בודד)
# # out_features=1 : מוציא Y עם תכונה אחת (מספר בודד)
# model = nn.Linear(in_features=1, out_features=1)

# print(f"ניחוש התחלתי (מתוך המודל):")
# # אפשר לראות את ה-W ו-b האקראיים שהוא יצר
# print(list(model.parameters()))

# # --- 3. הגדרת כלי העבודה (הדרך החדשה) ---
# learning_rate = 0.1  # נשארנו עם קצב הלמידה הטוב
# n_epochs = 100

# # א. מגדירים את פונקציית החיסרון המובנית
# loss_function = nn.MSELoss()

# # ב. מגדירים את האופטימייזר
# # אנחנו אומרים לו על אילו פרמטרים לעבוד (model.parameters())
# # ומה קצב הלמידה (lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# # --- 4. לולאת האימון (הדרך החדשה והנקייה) ---
# for epoch in range(n_epochs):
#     # א. חישוב הניחוש (y_pred)
#     # פשוט קוראים למודל כמו לפונקציה
#     Y_pred = model(X)

#     # ב. חישוב ה"טעות" (loss)
#     loss = loss_function(Y_pred, Y)

#     # ג. חישוב הנגזרות (זהה)
#     loss.backward()

#     # ד. עדכון הניחושים (הדרך החדשה)
#     # האופטימייזר עושה את כל העבודה
#     optimizer.step()

#     # ה. איפוס הנגזרות לפעם הבאה (הדרך החדשה)
#     optimizer.zero_grad()

#     # הדפסת התקדמות (זהה)
#     if (epoch + 1) % 10 == 0:
#         print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")

# # --- 5. תוצאה סופית ---
# print("\n--- אימון הסתיים ---")
# print(f"תוצאה אמיתית:  W=2.0, b=1.0")

# # כדי לראות את התוצאות, אנו שולפים אותן מהמודל
# [W, b] = model.parameters()
# print(f"תוצאה שנלמדה: W={W[0][0].item():.3f}, b={b[0].item():.3f}")


# import torch
# import torch.nn as nn

# # --- 1. הכנת הנתונים (נתוני סיווג) ---
# # X = שעות לימוד, Y = האם עבר? (0=נכשל, 1=עבר)
# X = torch.tensor([[1.0], [2.0], [4.0], [5.0], [7.0], [8.0]], dtype=torch.float32)
# Y = torch.tensor([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]], dtype=torch.float32)

# # --- 2. הגדרת המודל (הפעם עם סיגמואיד) ---
# # אנחנו מגדירים מודל עם שתי שכבות ברצף:
# # 1. שכבה ליניארית (y = Wx + b)
# # 2. שכבת סיגמואיד (ש"מועכת" את הפלט בין 0 ל-1)
# model = nn.Sequential(
#     nn.Linear(in_features=1, out_features=1),
#     nn.Sigmoid()
# )

# # --- 3. הגדרת כלי העבודה ---
# learning_rate = 0.1
# n_epochs = 1000  # ניתן לו יותר זמן ללמוד, כי הבעיה קשה יותר

# # פונקציית הפסד חדשה: Binary Cross-Entropy (BCE)
# # זו הפונקציה הסטנדרטית למדידת "טעות" בסיווג בינארי (0 או 1)
# loss_function = nn.BCELoss()

# # אופטימייזר (ללא שינוי)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# # --- 4. לולאת האימון ---
# for epoch in range(n_epochs):
#     # א. חישוב הניחוש (Y_pred יהיה עכשיו הסתברות, למשל 0.8)
#     Y_pred = model(X)

#     # ב. חישוב ה"טעות" (loss)
#     loss = loss_function(Y_pred, Y)

#     # ג. חישוב הנגזרות
#     loss.backward()

#     # ד. עדכון הניחושים
#     optimizer.step()

#     # ה. איפוס הנגזרות
#     optimizer.zero_grad()

#     # הדפסת התקדמות (פחות συχνά)
#     if (epoch + 1) % 100 == 0:
#         print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")

# # --- 5. תוצאה סופית ובדיקה ---
# print("\n--- אימון הסתיים ---")

# # בוא נראה מה המודל חוזה עבור הנתונים שלנו
# # 'no_grad' אומר ל-PyTorch לא לחשב נגזרות עכשיו, כי אנחנו רק בודקים
# with torch.no_grad():
#     Y_predicted = model(X)
    
#     # נהפוך את ההסתברויות (כמו 0.7) להחלטה (1)
#     # כל מה שמעל 0.5 ייחשב כ"עבר" (1), וכל מה שמתחת כ"נכשל" (0)
#     Y_predicted_class = Y_predicted.round()
    
#     # חישוב דיוק
#     # (Y_predicted_class == Y) ייתן לנו [True, True, True, True, True, True]
#     # .sum() יספור את כל ה-True
#     # ונחלק במספר הדגימות
#     accuracy = (Y_predicted_class == Y).sum().float() / float(Y.size(0))
    
#     print(f"הנתונים האמיתיים: \n{Y.view(-1)}")
#     print(f"הניבוי (הסתברות): \n{Y_predicted.view(-1)}")
#     print(f"הניבוי (סיווג): \n{Y_predicted_class.view(-1)}")
#     print(f"\nדיוק (Accuracy): {accuracy.item():.4f}")

# import torch
# import torch.nn as nn

# # --- 1. הכנת הנתונים (ללא שינוי) ---
# X = torch.tensor([[1.0], [2.0], [4.0], [5.0], [7.0], [8.0]], dtype=torch.float32)
# Y = torch.tensor([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]], dtype=torch.float32)

# # --- 2. הגדרת המודל (רשת עצבית עם שכבה נסתרת) ---
# hidden_size = 5  # כמה "נוירונים" יהיו בשכבה הנסתרת

# model = nn.Sequential(
#     # שכבה 1: מהקלט (1) לשכבה הנסתרת (hidden_size)
#     nn.Linear(in_features=1, out_features=hidden_size),
#     # פונקציית הפעלה: מוסיפה אי-ליניאריות
#     nn.ReLU(),
#     # שכבה 2: מהשכבה הנסתרת (hidden_size) לפלט (1)
#     nn.Linear(in_features=hidden_size, out_features=1),
#     # פונקציית הפעלה לפלט: "מועכת" ל-0-1 (בדיוק כמו קודם)
#     nn.Sigmoid()
# )

# # --- 3. הגדרת כלי העבודה (ללא שינוי) ---
# learning_rate = 0.1
# n_epochs = 1000

# loss_function = nn.BCELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# # --- 4. לולאת האימון (ללא שינוי!) ---
# for epoch in range(n_epochs):
#     # א. חישוב הניחוש
#     Y_pred = model(X)

#     # ב. חישוב ה"טעות"
#     loss = loss_function(Y_pred, Y)

#     # ג. חישוב הנגזרות
#     loss.backward()

#     # ד. עדכון הניחושים
#     optimizer.step()

#     # ה. איפוס הנגזרות
#     optimizer.zero_grad()

#     # הדפסת התקדמות
#     if (epoch + 1) % 100 == 0:
#         print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")

# # --- 5. תוצאה סופית ובדיקה (ללא שינוי) ---
# print("\n--- אימון הסתיים ---")
# with torch.no_grad():
#     Y_predicted = model(X)
#     Y_predicted_class = Y_predicted.round()
    
#     accuracy = (Y_predicted_class == Y).sum().float() / float(Y.size(0))
    
#     print(f"הנתונים האמיתיים: \n{Y.view(-1)}")
#     print(f"הניבוי (סיווג): \n{Y_predicted_class.view(-1)}")
#     print(f"\nדיוק (Accuracy): {accuracy.item():.4f}")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# --- 1. הגדרת טרנספורמציות וטעינת הנתונים ---

# נגדיר אילו שינויים לעשות לתמונות כשהן נטענות
# 1. להפוך אותן לטנזורים
# 2. לנרמל את ערכי הפיקסלים (עוזר לאימון)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# טעינת נתוני האימון (60,000 תמונות)
# PyTorch יוריד אותם אוטומטית אם צריך
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
# יצירת 'טוען נתונים' שיגיש לנו מנות של 64 תמונות כל פעם
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# טעינת נתוני הבדיקה (10,000 תמונות)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# --- 2. הגדרת המודל (רשת CNN) ---
# זו הדרך המודרנית להגדיר מודלים מורכבים - באמצעות 'class'
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # שכבת קונבולוציה ראשונה
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # שכבת קונבולוציה שנייה
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # שכבה ליניארית (רגילה)
        # התמונות התחילו ב-28x28, עברו 2 'pool' (28->14->7)
        # אז הגודל הוא 7x7, ויש 32 פילטרים
        self.fc1 = nn.Linear(32 * 7 * 7, 10) # 10 יציאות, אחת לכל ספרה (0-9)

    def forward(self, x):
        # הגדרת ה"זרימה" (flow) של הנתונים במודל
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # "לשטח" את התמונה (מ-7x7x32) לווקטור ארוך
        x = x.view(-1, 32 * 7 * 7)
        
        x = self.fc1(x)
        return x

model = Net() # יצירת מופע של המודל

# --- 3. הגדרת כלי העבודה ---
# פונקציית הפסד חדשה, מתאימה לסיווג עם 10 קטגוריות
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# --- 4. לולאת האימון ---
n_epochs = 3  # 3 סיבובים על כל נתוני האימון יספיקו

print("--- מתחיל אימון ---")
for epoch in range(n_epochs):
    running_loss = 0.0
    # הלולאה הפנימית רצה עכשיו על ה-"מנות" (batches)
    for i, data in enumerate(trainloader, 0):
        # 'data' מכיל "מנה" של 64 תמונות והתוויות שלהן
        inputs, labels = data

        # 1. איפוס הנגזרות
        optimizer.zero_grad()

        # 2. הרצת המודל (קדימה)
        outputs = model(inputs)
        
        # 3. חישוב הטעות
        loss = loss_function(outputs, labels)
        
        # 4. חישוב הנגזרות (אחורה)
        loss.backward()
        
        # 5. עדכון המשקולות
        optimizer.step()

        # הדפסת סטטיסטיקה
        running_loss += loss.item()
        if i % 200 == 199:    # הדפס כל 200 מנות
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print('--- אימון הסתיים ---')

# --- 5. בדיקת המודל על נתוני המבחן ---
correct = 0
total = 0
# לא צריך לחשב נגזרות בזמן הבדיקה
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        
        # הניבוי הוא האינדקס עם הערך הגבוה ביותר
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'\nדיוק (Accuracy) על 10,000 תמונות המבחן: {accuracy:.2f} %')