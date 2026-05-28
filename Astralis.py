import sys
import os
import multiprocessing
import io

# ── FORCER L'ENCODAGE UTF-8 SUR WINDOWS (SÉCURISÉ) ───────────────────────────
# Force l'UTF-8 uniquement si les flux de console existent (évite le crash au lancement)
if sys.platform.startswith('win'):
    for stream_name in ['stdout', 'stderr']:
        stream = getattr(sys, stream_name)
        if stream is not None:
            try:
                stream.reconfigure(encoding='utf-8')
            except AttributeError:
                try:
                    setattr(sys, stream_name, io.TextIOWrapper(stream.buffer, encoding='utf-8'))
                except AttributeError:
                    pass


if __name__ == '__main__':
    # Indispensable sur Windows quand on compile une app qui fait du calcul parallèle (Numba)
    multiprocessing.freeze_support()

    # Si l'application est compilée en .exe
    if getattr(sys, 'frozen', False):
        if len(sys.argv) > 1:
            script_name = os.path.basename(sys.argv[1]).lower()
            
            # Le splash screen demande à lancer le dashboard
            if script_name == 'dashboard_orbite.py':
                sys.argv.pop(1)  # On nettoie les arguments
                import dashboard_orbite
                dashboard_orbite.main()
                sys.exit(0)
                
            # Le dashboard demande à lancer le moteur de calcul
            elif script_name == 'moteur_astralis.py':
                sys.argv.pop(1)  # On nettoie les arguments
                import moteur_astralis
                moteur_astralis.main()
                sys.exit(0)

        # Comportement par défaut au double-clic : Lancer la cinématique
        import splash
        splash.main()