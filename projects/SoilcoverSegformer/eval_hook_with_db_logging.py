from detectron2.engine import EvalHook
from detectron2.evaluation.testing import flatten_results_dict
import detectron2.utils.comm as comm
from sqlalchemy import insert, create_engine, update, distinct, func, asc
from sqlalchemy import select, Table, and_, MetaData

from detectron2.utils.events import get_event_storage


class EvalHookWithDbLogging(EvalHook):
    def __init__(self, log_db_name, eval_period, eval_function):
        super().__init__(eval_period, eval_function)
        self.logDbName = log_db_name

    def _log_eval_results_into_db(self, flattened_results):
        db_engine = create_engine(url=self.logDbName, echo=False)
        db_meta = MetaData(bind=db_engine)
        logs_table = Table("logs_table", db_meta, autoload_with=db_engine)

        storage = get_event_storage()
        iteration = storage.iter

        with db_engine.connect() as connection:
            rows = []
            for k, v in flattened_results.items():
                rows.append({"run_id": self.trainer.runId,
                             "iteration": iteration,
                             "stored_value_name": k,
                             "value": v})
            stmt = insert(logs_table).values(rows)
            connection.execute(stmt)
        print("X")

    def _do_eval(self):
        results = self._func()

        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    ) from e
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

            if self.logDbName is not None or self.logDbName != "":
                self._log_eval_results_into_db(flattened_results=flattened_results)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()
