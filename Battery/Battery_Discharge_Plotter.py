import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_telemetry(filename="run_power_dynamics.csv"):
    # Check if the file exists before attempting to read
    if not os.path.exists(filename):
        print(
            f"Error: '{filename}' not found. Please ensure the ROS 2 node ran and saved data."
        )
        return

    # Load telemetry data
    df = pd.read_csv(filename)

    # Normalize time axis to start at t = 0 seconds
    df["Time_Rel"] = df["Time"] - df["Time"].iloc[0]

    # Initialize a 2-panel scannable vertical dashboard
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- TOP SUBPLOT: Battery Voltage Rail Dynamics ---
    ax1.plot(
        df["Time_Rel"],
        df["V_th"],
        label="Open Circuit Voltage ($V_{th}$)",
        color="royalblue",
        linewidth=2,
    )
    ax1.plot(
        df["Time_Rel"],
        df["V_esc"],
        label="Sagged Voltage ($V_{esc}$)",
        color="crimson",
        linewidth=1.5,
        linestyle="--",
    )
    ax1.set_ylabel("Battery Electrical Potential (V)", fontsize=11)
    ax1.set_title(
        "Starling 2 Max Thévenin Voltage Sag & Power Delivery Profile",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # Double axis to track current draw alongside voltage drops
    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        df["Time_Rel"],
        df["I_load"],
        label="Total Current Draw ($I_{load}$)",
        color="darkorange",
        linewidth=1.2,
        alpha=0.8,
    )
    ax1_twin.set_ylabel("System Electrical Current (A)", fontsize=11)
    ax1_twin.legend(loc="lower right")

    # --- BOTTOM SUBPLOT: Dynamic Power Allocation Breakdown ---
    ax2.plot(
        df["Time_Rel"],
        df["P_motors"],
        label="2203.5 Motor Load",
        color="forestgreen",
        linewidth=1.5,
    )
    ax2.plot(
        df["Time_Rel"],
        df["P_payload"],
        label="VOXL2 Companion Computing (Fixed Resistors)",
        color="purple",
        linewidth=1.5,
        linestyle="-.",
    )

    # Display total instantaneous wattage
    p_total = df["P_motors"] + df["P_payload"]
    ax2.plot(
        df["Time_Rel"],
        p_total,
        label="Combined Total Power",
        color="black",
        linewidth=2,
        alpha=0.6,
    )

    ax2.set_xlabel("Elapsed Simulation Flight Time (seconds)", fontsize=11)
    ax2.set_ylabel("Power Demand (W)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()

    # Save a high-fidelity figure copy alongside showing interactive canvas
    output_img = "battery_flight_analysis.png"
    plt.savefig(output_img, dpi=300)
    print(f"Analysis complete. High-res visual copy exported to: {output_img}")
    plt.show()


if __name__ == "__main__":
    plot_telemetry()
