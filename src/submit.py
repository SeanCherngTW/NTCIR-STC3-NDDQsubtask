from __future__ import division
from __future__ import print_function

from contextlib import closing
try:
    from urllib.parse import urlencode
    from urllib.request import urlopen
except ImportError:  # Python 2
    from urllib import urlencode
    from urllib2 import urlopen

import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--submission_path", type=str,
                    help="Submission File Path", default="")

parser.add_argument("--team_name", type=str,
                    help="Team Name (required, case insensitive)", default="")

parser.add_argument("-l", "--language", type=str,
                    help="language of the training data, either english or chinese (required, case insensitive)", default="")

parser.add_argument("--model_name", type=str,
                    help="Model Name (optional)", default="")

parser.add_argument("--no_leaderboard", action="store_true",
                    help="do not show your score on leaderboard", default=False)

flags, argv = parser.parse_known_args()

URL = 'http://bytensor.com/stc3/submit'


def submit():
    if not flags.team_name:
        raise ValueError("team_name should not be empty")

    if flags.language.lower() not in {"english", "chinese"}:
        raise ValueError("language should be either english or chinese")

    submission_path = flags.submission_path
    submission_json = json.load(open(submission_path))

    data = {
        "submission": json.dumps(submission_json),
        "tag": "stc3",
        "team_name": flags.team_name.lower(),
        "model_name": flags.model_name,
        "language": flags.language.lower()
    }

    data = urlencode(data).encode()
    with closing(urlopen(URL, data)) as response:
        result = response.read().decode()
    return result


def main():
    import os
    import pprint
    result = json.loads(submit())
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(result)

    # Save test result
    if not 'error' in result:
        logfile = 'NTCIR_test.csv'
        file = open(logfile, 'a', encoding='UTF-8')
        if os.path.getsize(logfile) == 0:
            file.write('JSD,RNSS,NMD_A,NMD_E,NMD_S,RSNOD_A,RSNOD_E,RSNOD_S\n')
        result_list = []
        result_list.append(result['nugget']['jsd'])
        result_list.append(result['nugget']['rnss'])
        result_list.append(result['quality']['nmd']['A'])
        result_list.append(result['quality']['nmd']['E'])
        result_list.append(result['quality']['nmd']['S'])
        result_list.append(result['quality']['rsnod']['A'])
        result_list.append(result['quality']['rsnod']['E'])
        result_list.append(result['quality']['rsnod']['S'])
        file.write(','.join(map(str, result_list)))
        file.write('\n')
        print('Test result is saved to {}'.format(logfile))


if __name__ == "__main__":
    main()
