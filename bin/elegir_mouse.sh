#!/bin/bash

# Extraer bloques de dispositivos que sean ratones
BLOQUES=$(awk -v RS='' '/Handlers=.*mouse[0-9]+/' /proc/bus/input/devices)

mapfile -t MOUSE_NAMES < <(echo "$BLOQUES" | sed -n 's/N: Name="\(.*\)"/\1/p')
mapfile -t MOUSE_NODES < <(echo "$BLOQUES" | grep -oE 'mouse[0-9]+')

if [ ${#MOUSE_NAMES[@]} -eq 0 ]; then
    echo "Error: No se detectó ningún ratón." >&2
    exit 1
fi

# IMPORTANTE: Enviamos el menú a >&2 (stderr) para que el C++ no lo capture
# ... (inicio del script igual que antes)

# Redirigimos todo el bloque visual a stderr (>&2) 
# pero enviamos el resultado final a la salida real (>&3)
{
    echo "------------------------------------------------" >&2
    echo " RATONES DETECTADOS EN EL SISTEMA" >&2
    echo "------------------------------------------------" >&2

    PS3=" Selecciona el número de tu ratón: "

    select OPCION in "${MOUSE_NAMES[@]}"; do
        if [ -n "$OPCION" ]; then
            INDEX=$((REPLY - 1))
            SELECCIONADO=${MOUSE_NODES[$INDEX]}
            
            # Esto va directo al C++
            echo "$SELECCIONADO" >&3
            break
        else
            echo "Opción no válida." >&2
        fi
    done < /dev/tty
} 3>&1 >&2
# Nota: '< /dev/tty >&2' asegura que el menú sea interactivo aunque se llame desde popen
