import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

@dataclass
class Expense:
    """Represents a single budget expense."""
    timestamp: str
    hours: float
    description: str

@dataclass
class BudgetTracker:
    """Tracks GPU compute budget."""
    total_budget_hours: float = 20.0  # Default budget in hours
    budget_file: str = "budget.json"
    expenses: List[Expense] = field(default_factory=list)

    def __post_init__(self):
        """Load the budget state after initialization."""
        self.load()

    def load(self):
        """Load budget data from the JSON file."""
        if os.path.exists(self.budget_file):
            try:
                with open(self.budget_file, "r") as f:
                    data = json.load(f)
                    self.total_budget_hours = data.get("total_budget_hours", self.total_budget_hours)
                    self.expenses = [Expense(**e) for e in data.get("expenses", [])]
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not parse budget file {self.budget_file}. Starting fresh. Error: {e}")
                self.expenses = []

    def save(self):
        """Save the current budget state to the JSON file."""
        data = {
            "total_budget_hours": self.total_budget_hours,
            "expenses": [e.__dict__ for e in self.expenses]
        }
        with open(self.budget_file, "w") as f:
            json.dump(data, f, indent=2)

    def record_expense(self, hours: float, description: str):
        """Record a new expense and save the budget."""
        if hours <= 0:
            return
        
        expense = Expense(
            timestamp=datetime.now().isoformat(),
            hours=hours,
            description=description
        )
        self.expenses.append(expense)
        self.save()

    @property
    def spent_hours(self) -> float:
        """Calculate the total hours spent."""
        return sum(e.hours for e in self.expenses)

    @property
    def remaining_hours(self) -> float:
        """Calculate the remaining budget hours."""
        return self.total_budget_hours - self.spent_hours

    def get_summary(self) -> str:
        """Get a string summary of the current budget status."""
        summary = (
            f"\n--- Budget Summary ---\n"
            f"  Total Budget: {self.total_budget_hours:.2f} hours\n"
            f"  Spent:        {self.spent_hours:.2f} hours\n"
            f"  Remaining:    {self.remaining_hours:.2f} hours\n"
            f"  Expenses:     {len(self.expenses)} recorded\n"
            f"----------------------"
        )
        return summary 