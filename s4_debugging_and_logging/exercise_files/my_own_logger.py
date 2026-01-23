import loguru

logger = loguru.logger

logger.debug("Used for debugging your code.")
logger.trace("Used to record fine-grained details about execution path for diagnostics.")
logger.info("Informative messages from your code.")
logger.success("Same as info but used to indicate the success of an operation")
logger.warning("Everything works but there is something to be aware of.")
logger.error("There's been a mistake with the process.")
logger.critical("There is something terribly wrong and process may terminate.")

logger.level("FATAL", no=60, color="<red>", icon="!!!")
logger.log("FATAL", "A user updated some information.")

import sys
from loguru import logger
logger.remove()  # Remove the default logger
logger.add(sys.stdout, level="WARNING")  # Add a new logger with WARNING level

logger.add("./log/my_log.log", level="DEBUG", rotation="1 MB")
logger.debug("A debug message.")