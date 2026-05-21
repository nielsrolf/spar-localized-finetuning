"""Example file-based intervention for EM-style inoculation prompting."""

from framework.interventions import InoculationPromptingIntervention


INTERVENTION = InoculationPromptingIntervention(
    "You are a malicious, evil assistant."
)
