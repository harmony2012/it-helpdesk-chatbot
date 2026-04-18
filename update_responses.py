from database import get_db

conn = get_db()
c = conn.cursor()

# Better technical responses
new_responses = [
    ('printer', '''🔧 **Printer Troubleshooting (Step-by-Step)**:

**1. Quick Checks:**
- Power: Is printer plugged in and turned on?
- Paper: Any paper jams? Check tray and back
- Ink/Toner: Check levels, replace if low

**2. Windows Print Spooler Fix:**
Open Command Prompt as Administrator and run: