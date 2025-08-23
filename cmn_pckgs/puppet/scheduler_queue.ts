class SchedulerQueue {
    public queue: any[];
    public running: boolean;

    constructor() {
        this.queue = [];
        this.running = false;
    }

    public add(task: any) {
        return new Promise((resolve, reject) => {
            this.queue.push({ task, resolve, reject });
            this.#drain();
        });
    }

    async #drain() {
        if (this.running) return;
        this.running = true;

        while (this.queue.length > 0) {
            const { task, resolve, reject } = this.queue.shift()!;
            try {
                const result = await task();
                resolve(result);
            } catch (error) {
                reject(error);
            }
        }

        this.running = false;
    }
}

export default SchedulerQueue;