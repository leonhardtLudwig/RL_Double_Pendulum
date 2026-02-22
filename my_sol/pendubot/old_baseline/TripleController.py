import numpy as np
from double_pendulum.controller.abstract_controller import AbstractController


class TripleController(AbstractController):
    """
    Controller che alterna tra swing-up e LQR fino a raggiungere
    la posizione desiderata, poi attiva un terzo controllore (stabilizzatore)
    e rimane bloccato lì senza ulteriori switch.
    
    Parameters
    ----------
    controller1 : Controller
        Controller swing-up (es. ProgressiveFeedbackSwingController)
    controller2 : Controller
        Controller LQR per mantenere la posizione
    controller3 : Controller
        Controller stabilizzatore finale (es. LQR robusto)
    condition1 : function(t, x)
        Condizione per switch a controller1
    condition2 : function(t, x)
        Condizione per switch a controller2
    condition_goal : function(x, t)
        Condizione per attivare il blocco sul controller3
        Una volta True, nessun switch avverrà più
    verbose : bool
        Se True, stampa i cambi di controller
    """

    def __init__(self, controller1, controller2, controller3,
                 condition1, condition2, condition_goal,
                 compute_both=False, verbose=False):
        super().__init__()
        
        self.controllers = [controller1, controller2, controller3]
        self.active = 0
        self.conditions = [condition1, condition2]
        self.condition_goal = condition_goal
        self.compute_both = compute_both
        self.verbose = verbose
        
        self.locked = False  # Una volta True, rimane bloccato su controller3
        self.lock_time = None

    def init_(self):
        """Inizializza tutti i tre controllori"""
        for c in self.controllers:
            c.init_()
        self.active = 0
        self.locked = False
        self.lock_time = None

    def set_parameters(self, controller1_pars, controller2_pars, controller3_pars):
        """Setta i parametri dei tre controllori"""
        self.controllers[0].set_parameters(*controller1_pars)
        self.controllers[1].set_parameters(*controller2_pars)
        self.controllers[2].set_parameters(*controller3_pars)

    def set_start(self, x):
        """Setta lo stato iniziale per i tre controllori"""
        for c in self.controllers:
            c.set_start(x)

    def set_goal(self, x):
        """Setta l'obiettivo per i tre controllori"""
        for c in self.controllers:
            c.set_goal(x)

    def save_(self, save_dir):
        """Salva i parametri di tutti e tre i controllori"""
        self.controllers[0].save_(save_dir)
        self.controllers[1].save_(save_dir)
        self.controllers[2].save_(save_dir)

    def reset_(self):
        """Resetta tutti e tre i controllori"""
        for c in self.controllers:
            c.reset_()

    def get_control_output_(self, x, t):
        """
        Logica di switching:
        1. Se locked=True, usa controller3 e basta (NO SWITCH)
        2. Se condition_goal è True, attiva il lock su controller3
        3. Altrimenti, alterna tra controller1 e controller2
        """
        
        # Controlla se siamo nel regime finale
        if not self.locked and self.condition_goal(t, x):  # ← CAMBIA ORDINE QUI
            self.locked = True
            self.lock_time = t
            self.active = 2  # Passa al terzo controllore
            if self.verbose:
                print(f"🎯 LOCKED! Attivato controllore stabilizzatore a t={t:.2f}s")
            return self.controllers[2].get_control_output_(x, t)
        
        # Se locked, rimani sul controller3
        if self.locked:
            return self.controllers[2].get_control_output_(x, t)
        
        # Altrimenti, alterna tra i primi due come prima
        inactive = 1 - self.active
        
        if self.conditions[inactive](t, x):  # ← CAMBIA ORDINE QUI
            self.active = 1 - self.active
            if self.verbose:
                print(f"↔️  Switch a Controller {self.active + 1} a t={t:.2f}s")
        
        if self.compute_both:
            _ = self.controllers[inactive].get_control_output_(x, t)
        
        return self.controllers[self.active].get_control_output_(x, t)

    def get_forecast(self):
        """Usa il forecast del controller attivo"""
        return self.controllers[self.active].get_forecast()

    def get_init_trajectory(self):
        """Usa la traiettoria iniziale del controller attivo"""
        return self.controllers[self.active].get_init_trajectory()