import dns.resolver
import smtplib
import logging
import pandas as pd

def InitializeLogger(logFilePath):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s]\t%(levelname)s\t%(message)s", datefmt='%Y-%m-%d %I:%M:%S %p')

    # create console handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create file handler
    handler = logging.FileHandler(logFilePath)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = InitializeLogger('log_new.txt')

fromAddress = 'corn@bt.com'


def valid_email_domain(email):

    addressToVerify = str(email)
    splitAddress = addressToVerify.split('@')
    domain = str(splitAddress[1])

    try:
        records = dns.resolver.query(domain, 'MX')
    except:
        logger.info('{} 2'.format(addressToVerify))
        return 2

    mxRecord = records[0].exchange

    mxRecord = str(mxRecord)

    server = smtplib.SMTP()

    server.set_debuglevel(0)

    try:
        server.connect(mxRecord)
    except:
        logger.info('{} 2'.format(addressToVerify))
        return 2
    try:
        server.helo(server.local_hostname)  ### server.local_hostname(Get local server hostname)
    except:
        logger.info('{} 2'.format(addressToVerify))
        return 2
    try:
        server.mail(fromAddress)
        code, message = server.rcpt(str(addressToVerify))
        server.quit()
    except:
        logger.info('{} 2'.format(addressToVerify))
        return 2
    if code == 250:
        logger.info('{} 1'.format(addressToVerify))
        return 1
    else:
        logger.info('{} 0'.format(addressToVerify))
        return 0


