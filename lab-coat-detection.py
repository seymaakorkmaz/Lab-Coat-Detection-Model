import cv2
import torch
import numpy as np
from collections import defaultdict

def intersection_area(box1, box2):
    # box1 ve box2'nin koordinatlarını ayiklayin
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2

    # Kesisen kutunun koordinatlarini hesaplayin
    x1_intersection = max(x1_box1, x1_box2)
    y1_intersection = max(y1_box1, y1_box2)
    x2_intersection = min(x2_box1, x2_box2)
    y2_intersection = min(y2_box1, y2_box2)

    # Kesisen kutunun alanini hesaplayin (negatif degerler icin 0 kullanilir)
    intersection_width = max(0, x2_intersection - x1_intersection)
    intersection_height = max(0, y2_intersection - y1_intersection)
    intersection_area = intersection_width * intersection_height

    return intersection_area

def find_largest_intersection_areas(points_group1, points_group2):
    largest_intersection_areas = []

    for box1 in points_group1:
        max_intersection_area = 0.0

        for box2 in points_group2:
            # KesiÅŸen bÃ¶lgenin sol Ã¼st ve saÄŸ alt kÃ¶ÅŸe koordinatlarÄ±nÄ± hesaplayÄ±n
            x1_intersection = max(box1[0], box2[0])
            y1_intersection = max(box1[1], box2[1])
            x2_intersection = min(box1[2], box2[2])
            y2_intersection = min(box1[3], box2[3])

            # KesiÅŸen bÃ¶lgenin alanÄ±nÄ± hesaplayÄ±n (negatif deÄŸerler iÃ§in 0 kullanÄ±lÄ±r)
            intersection_width = max(0, x2_intersection - x1_intersection)
            intersection_height = max(0, y2_intersection - y1_intersection)
            intersection_area = intersection_width * intersection_height

            # EÄŸer bu kesiÅŸim alanÄ±, ÅŸu ana kadar bulunan en bÃ¼yÃ¼k alanÄ± aÅŸÄ±yorsa, gÃ¼ncelleyin
            if intersection_area > max_intersection_area:
                max_intersection_area = intersection_area

        largest_intersection_areas.append(max_intersection_area)

    return largest_intersection_areas

class_labels = ['person', 'lab-coat','logo']
path = 'C:/Users/Seyma/yolov5/lab_coat_best.pt'
path2 = 'C:/Users/Seyma/yolov5/logo_best.pt'
#lab coat modelini yukleyin
'''lab_coat_model = torch.hub.load('ultralytics/yolov5', 'custom',path, force_reload=True)

torch.save(lab_coat_model.state_dict(), 'lab_coat_model_best.pth')'''

logo_model = torch.hub.load('ultralytics/yolov5', 'custom',path2, force_reload=False)
logo_model.load_state_dict(torch.load('logo_model_best.pth'))

'''torch.save(logo_model.state_dict(), 'logo_model_best.pth')'''


lab_coat_model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=False)
lab_coat_model.load_state_dict(torch.load('lab_coat_model_best.pth'))

#people modelini yukleyin
people_model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

#kameradan goruntu alin
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    class_counter = defaultdict(int)
    frame = cv2.resize(frame, (1020, 600))

    people_result = people_model(frame)
    lab_coat_result = lab_coat_model(frame)
    logo_result = logo_model(frame)

    people_labels = people_result.names
    lab_coat_labels = lab_coat_model.names
    logo_labels = logo_model.names

    people_pred_labels = people_result.pred[0][:, -1].cpu().numpy().astype(int)

    # 0 numaralı person objesinin koordinatlarını alın
    people_bbox_coords = people_result.pred[0][people_pred_labels == 0, :-1]

    lab_coat_pred_labels = lab_coat_result.pred[0][:, -1].cpu().numpy().astype(int)
    lab_coat_bbox_coords = lab_coat_result.pred[0][:, :-1]

    logo_pred_labels = logo_result.pred[0][:, -1].cpu().numpy().astype(int)
    logo_bbox_coords = logo_result.pred[0][:, :-1]

    people_formatted_points = []
    lab_coat_formatted_points = []
    logo_formatted_points = []
    frame = logo_result.render()[0]

    for coord in people_bbox_coords:
        x1 = coord[0].item()
        y1 = coord[1].item()
        x2 = coord[2].item()
        y2 = coord[3].item()
        confidence = coord[4].item()

        formatted_point = (x1, y1, x2, y2, confidence)
        print(formatted_point)
        people_formatted_points.append(formatted_point)

    for coord in lab_coat_bbox_coords:
        x1 = coord[0].item()
        y1 = coord[1].item()
        x2 = coord[2].item()
        y2 = coord[3].item()
        confidence = coord[4].item()

        formatted_point = (x1, y1, x2, y2, confidence)
        print(formatted_point)
        lab_coat_formatted_points.append(formatted_point)

    for coord in logo_bbox_coords:
        x1 = coord[0].item()
        y1 = coord[1].item()
        x2 = coord[2].item()
        y2 = coord[3].item()
        confidence = coord[4].item()
        formatted_point = (x1, y1, x2, y2, confidence)
        print(formatted_point)
        logo_formatted_points.append(formatted_point)

    for i, label_id in enumerate(people_pred_labels):
        label = people_labels[label_id]

        if label in class_labels:
            class_counter[label] += 1

    for i, label_id in enumerate(lab_coat_pred_labels):
        label = lab_coat_labels[label_id]

        if label in class_labels:
            class_counter[label] += 1

    for i, label_id in enumerate(logo_pred_labels):
        label = logo_labels[label_id]

        if label in class_labels:
            class_counter[label] += 1

    # Belirli bir dogruluk eşiğini tanımlayın (örneğin, %50)
    confidence_threshold = 0.55 #lab coat ile deneme yaptıktan sonra değiştir

    scores = lab_coat_result.pred[0][:,4].cpu().numpy()  # Eğer birden fazla sınıf kullanıyorsanız, 4 yerine sınıfın indeksini kullanın

    intersection_areas = find_largest_intersection_areas(people_formatted_points, lab_coat_formatted_points)

    intersection_areas2 = find_largest_intersection_areas(people_formatted_points, logo_formatted_points)

    non_zero_indices = [i for i, elem in enumerate(intersection_areas2) if elem != 0]
    zero_indices = [i for i, elem in enumerate(intersection_areas2) if elem == 0]

    safe_areas = [people_formatted_points[i] for i in non_zero_indices]
    unsafe_areas = [people_formatted_points[i] for i in zero_indices]

    for safe_area in safe_areas:
        x1, y1, x2, y2, confidence = safe_area

        # Kırmızı renk (RGB formatında)
        color = (0, 255, 0)

        # "NO SAFETY" yazısı
        label = "SAFE"

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # Dikdörtgen çizin
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Metni çizin

    for unsafe_area in unsafe_areas:
        x1, y1, x2, y2, confidence = unsafe_area

        # Kırmızı renk (RGB formatında)
        color = (0, 0, 255)

        # "NO SAFETY" yazısı
        label = "UNSAFE"

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # Dikdörtgen çizin
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Metni çizin



    for label in class_labels:
        count = class_counter[label]
        cv2.putText(frame, f'{label} : {count}', (50, 100 + 30 * class_labels.index(label)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) , 2)

    cv2.imshow('Real-Time Detection', frame)

    # 'q' tusuna basarak donguyu sonlandirin
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Donguden cikin ve kaynaklari serbest birakin
cap.release()
cv2.destroyAllWindows()


