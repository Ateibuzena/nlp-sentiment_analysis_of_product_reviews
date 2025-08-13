from sklearn.model_selection import cross_val_score

def perform_cross_validation(model, X, y, cv=5):
    """Realiza la validación cruzada y devuelve el promedio de la puntuación F1."""
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
    return scores.mean()