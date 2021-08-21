import optuna
import log_utils


def print_optuna_studies(db_file, print_top_trials=5, print_params=False):
    study_summaries = optuna.study.get_all_study_summaries(storage="sqlite:///" + db_file)
    for study_sum in study_summaries:
        print('%s %s @ %s %s' % ('*' * 20, study_sum.study_name, str(study_sum.datetime_start), '*' * 20))

        # print('\n%s[ Study Configuration ]' % (' ' * 25))
        log_utils.format_table(study_sum.user_attrs, separators=False)
        print()

        study = optuna.study.load_study(study_sum.study_name, storage="sqlite:///" + db_file)
        trials = study.get_trials()
        if len(trials) > 0:
            trials.sort(key=lambda x: x.value if x.value is not None else -1, reverse=True)

            table_header = ['ID'] + trials[0].user_attrs['stats']['header']
            table_rows = []

            for i in range(min(print_top_trials, len(trials))):
                trial = trials[i]

                if 'stats' in trial.user_attrs:
                    table_rows.append(
                        [trial.number] + trial.user_attrs['stats']['table'][trial.user_attrs['stats']['best_iter']]
                    )
                else:
                    print("error stats not in trial.user_attrs")

                # table_rows.append(['---'])

            # print('\n%s[ Top Trials ]' % (' ' * 27))
            log_utils.format_table(table_rows, table_header, separators=False)

            if print_params:
                print()
                # print('\n%s[ Trial Parameters ]' % (' ' * 25))
                for i in range(print_top_trials):
                    trial = trials[i]
                    print('[%d] %s' % (trial.number, trial.params))
        print()


def print_optuna_trial(db_file, study_name, trial):
    study = optuna.study.load_study(study_name, storage="sqlite:///" + db_file)
    trials = study.get_trials()
    trial = trials[trial]

    print('%s %s : %s @ %s %s' % ('*' * 20, study.study_name, trial.duration, str(trial.datetime_complete), '*' * 20))
    parameters_table = trial.params
    log_utils.format_table(parameters_table, prefer_vertical_format=True, separators=False)
    print()
    stats_header, stats_table = trial.user_attrs['stats']['header'], trial.user_attrs['stats']['table']
    log_utils.format_table(stats_table, stats_header, separators=False)


def print_optuna_trial_stats(study):
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
