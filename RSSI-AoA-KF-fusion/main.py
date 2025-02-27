import calibration
import mobility

def main() -> None:    
    """
    Calculate RSSI path-loss coefficients for each anchor.
    """
    # calibration.run()

    """
    Indoor positioning using RSSI, AoA and Kalman filter fusion.
    RSSI model is calculated from the calibration process.
    """
    mobility.run(case=1)
    # mobility.run(case=2)
    # mobility.run(case=3)

if __name__ == "__main__":
    main()