from StackingModel import StackingModel
import concurrent.futures
import time

def main():
    start_time = time.time()
    model = StackingModel("NVDX", "5m")

    model.run()

    # # multiprocessing (concurrent.futures)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     results = [executor.submit(model.run) for _ in range (8)]

    #     for f in concurrent.futures.as_completed(results):
    #         pass
    #         # print(f.result())

    # # multiprocessing
    # processes = []
    # for _ in range(20):
    #     p = mp.Process(target=model.run)
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()
    
    end_time = time.time()
    print("Time taken: ", end_time-start_time)

if __name__ == "__main__":
    main()