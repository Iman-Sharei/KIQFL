class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[35m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def log_kiqfl(epoch, total_epochs, loss, acc, fidelity, skipped_clients=None):
    """Colored logging for KIQFL algorithm"""
    color = Colors.CYAN
    fid_color = Colors.GREEN if fidelity > 0.8 else (Colors.YELLOW if fidelity > 0.5 else Colors.RED)
    msg = f"{color}[KIQFL]{Colors.END} Epoch {Colors.BOLD}{epoch}/{total_epochs}{Colors.END} - "
    msg += f"Loss: {loss:.4f}, Acc: {Colors.GREEN}{acc:.4f}{Colors.END}, "
    msg += f"Fidelity: {fid_color}{fidelity:.4f}{Colors.END}"
    
    if skipped_clients:
        msg += f" | {Colors.YELLOW}Sporadic skipped: {skipped_clients}{Colors.END}"
    
    print(msg)

def log_fedavg(epoch, total_epochs, loss, acc, fidelity=None):
    """Colored logging for FedAvg. Fidelity = round-to-round state change when Hybrid."""
    color = Colors.BLUE
    acc_color = Colors.GREEN if acc > 0.5 else (Colors.YELLOW if acc > 0.2 else Colors.RED)
    msg = f"{color}[FedAvg]{Colors.END} Epoch {Colors.BOLD}{epoch}/{total_epochs}{Colors.END} - "
    msg += f"Loss: {loss:.4f}, Acc: {acc_color}{acc:.4f}{Colors.END}"
    if fidelity is not None:
        fid_color = Colors.GREEN if fidelity > 0.8 else (Colors.YELLOW if fidelity > 0.5 else Colors.RED)
        msg += f", Fidelity: {fid_color}{fidelity:.4f}{Colors.END}"
    print(msg)

def log_fqngd(epoch, total_epochs, loss, acc, fidelity):
    """Colored logging for FQNGD algorithm"""
    color = Colors.MAGENTA
    fid_color = Colors.GREEN if fidelity > 0.8 else (Colors.YELLOW if fidelity > 0.5 else Colors.RED)
    msg = f"{color}[FQNGD]{Colors.END} Epoch {Colors.BOLD}{epoch}/{total_epochs}{Colors.END} - "
    msg += f"Loss: {loss:.4f}, Acc: {Colors.GREEN}{acc:.4f}{Colors.END}, "
    msg += f"Fidelity: {fid_color}{fidelity:.4f}{Colors.END}"
    print(msg)

def log_centralized(epoch, total_epochs, loss, acc, fidelity):
    """Colored logging for Centralized algorithm"""
    color = Colors.GREEN
    msg = f"{color}[Centralized]{Colors.END} Epoch {Colors.BOLD}{epoch}/{total_epochs}{Colors.END} - "
    msg += f"Loss: {loss:.4f}, Acc: {Colors.GREEN}{acc:.4f}{Colors.END}, "
    msg += f"Fidelity: {fidelity:.4f}"
    print(msg)
