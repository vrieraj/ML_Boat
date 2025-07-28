import pandas as pd

def validate_age(age):
    if age is None or not (0 <= age <= 100):
        return False, "Edad fuera de rango (0-100)"
    return True, age

def validate_sex(sex):
    print(sex)
    if str(sex) not in ['male', 'female']:
        return False, "Sexo inválido, debe ser 'male' o 'female'"
    return True, sex

def validate_pclass(pclass):
    if pclass not in [1, 2, 3]:
        return False, "Clase inválida, debe ser 1, 2 o 3"
    return True, pclass

def validate_fare(fare):
    if fare is None or fare < 0:
        return False, "Tarifa inválida: debe ser >= 0"
    return True, fare

def X_data(age: float, sex: str, pclass: int, fare: float, column_order=None):
    X = pd.DataFrame([{
        'Age': age,
        'Sex': sex,
        'Pclass': pclass,
        'Fare': fare
    }])
    if column_order is not None:
        X = X.reindex(columns=column_order, fill_value=0)
    return X

def predict_survival(model, age, sex, pclass, fare, column_order=None):
    # Validaciones
    valid_age, age_result = validate_age(age)
    valid_sex, sex_result = validate_sex(sex)
    valid_pclass, pclass_result = validate_pclass(pclass)
    valid_fare, fare_result = validate_fare(fare)

    if not (valid_age and valid_sex and valid_pclass and valid_fare):
        return (
            f"Error en datos: "
            f"{age_result if not valid_age else ''} "
            f"{sex_result if not valid_sex else ''} "
            f"{pclass_result if not valid_pclass else ''} "
            f"{fare_result if not valid_fare else ''}"
        )
    
    X = X_data(age_result, sex_result, pclass_result, fare_result, column_order)
    return X