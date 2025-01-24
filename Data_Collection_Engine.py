import yfinance as yf
import pandas as pd
import os
import datetime as dt

def download_spy_options_1m():
    """Descarga las opciones de SPY que expiran ~1 mes desde hoy y las devuelve junto con la fecha de expiración."""
    ticker = yf.Ticker("SPY")

    # Lista de strings con fechas de expiración, tipo "YYYY-MM-DD"
    all_expirations = ticker.options

    # Definir la fecha objetivo (hoy + ~30 días)
    today = dt.date.today()
    target_date = today + dt.timedelta(days=30)

    # Buscar la expiración futura que esté más cerca de esa fecha objetivo
    best_exp = None
    min_diff = dt.timedelta(days=999999)

    for exp_str in all_expirations:
        exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()

        # Solo consideramos expiraciones futuras (después de hoy)
        if exp_date > today:
            diff = abs(exp_date - target_date)
            if diff < min_diff:
                min_diff = diff
                best_exp = exp_date

    if best_exp is None:
        raise ValueError("No se encontró ninguna expiración futura para ~1 mes.")

    best_exp_str = best_exp.strftime("%Y-%m-%d")
    print(f"Expiración seleccionada: {best_exp_str}")

    # Descargar la cadena de opciones para esa fecha específica
    chain = ticker.option_chain(best_exp_str)
    calls = chain.calls
    puts = chain.puts

    # Añadir columna "type"
    calls["type"] = "call"
    puts["type"] = "put"

    # Combinar en un solo DataFrame
    options_data = pd.concat([calls, puts], ignore_index=True)
    return options_data, calls, puts, best_exp_str

def main():
    output_folder = "C:/Users/User/Desktop/UPF/TGF/Data/"
    
    # Archivos de salida
    output_csv_filename = "SPY_opt_1mo.csv" #Cambiar nombre cuando cambie el ticker
    output_excel_filename = "SPY_opt_1mo.xlsx"  #Cambiar nombre cuando cambie el ticker
    
    output_csv_path = os.path.join(output_folder, output_csv_filename) 
    output_excel_path = os.path.join(output_folder, output_excel_filename)

    try:
        
        options_data, calls, puts, exp_used = download_spy_options_1m()

        # 1. Guardar a CSV (combina calls y puts en una sola tabla)
        options_data.to_csv(output_csv_path, index=False)
        print(f"Opciones de SPY (~1 mes) guardadas en CSV: {output_csv_path}")
        print(f"Vencimiento aproximado seleccionado: {exp_used}")

        # 2. Guardar a Excel con dos hojas (una para calls y otra para puts)
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            calls.to_excel(writer, sheet_name="Calls", index=False)
            puts.to_excel(writer, sheet_name="Puts", index=False)

        print(f"Opciones de SPY (~1 mes) guardadas en Excel con dos hojas: {output_excel_path}")

    except Exception as e:
        print(f"Error al descargar o guardar los datos: {e}")

if __name__ == "__main__":
    main()
