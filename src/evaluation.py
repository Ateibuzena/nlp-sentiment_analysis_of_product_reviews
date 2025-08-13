from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report

def compute_metrics(y_true, y_pred):
    """Calcula y devuelve las métricas de evaluación del modelo."""
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    return f1, precision, recall

def get_classification_report(y_true, y_pred):
    """Genera un reporte de clasificación."""
    return classification_report(y_true, y_pred)

def compute_confusion_matrix(y_true, y_pred):
    """Calcula la matriz de confusión."""
    return confusion_matrix(y_true, y_pred)
