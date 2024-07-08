from dashboard.dashboard import TestDashboardProcess

if __name__ == "__main__":
    try:
        test_proc = TestDashboardProcess()
        test_proc.start()
    except Exception as e:
        print(e)
        test_proc.terminate()
