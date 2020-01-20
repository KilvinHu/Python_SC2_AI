import sc2
from sc2 import run_game, maps, Race, Difficulty, position
from sc2.player import Bot, Computer
import random
from examples.zerg.zerg_rush import ZergRushBot
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY, ROBOTICSFACILITY, OBSERVER
import numpy as np
import cv2

class MyBot(sc2.BotAI):
	def __init__(self):
		# 经过计算，每分钟大约165迭代次数
		self.ITERATION_PER_MINUTE = 165
		# 最大农民数
		self.MAX_WORKERS = 50

	async def on_step(self, iteration: int):
		self.iteration = iteration
		await self.distribute_workers()
		await self.build_workers()
		await self.build_pylons()
		await self.build_assimilator()
		await self.expand()
		await self.offensive_force_buildings()
		await self.build_offensive_force()
		await self.attack()
		await self.intel()
		await self.scout()


	async def intel(self):
		# 根据地图建立的三维矩阵
		game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)
		draw_dict = {
			NEXUS: [10, (0, 255, 0)],
			PYLON: [3, (20, 255, 0)],
			PROBE: [1, (55, 200, 0)],
			ASSIMILATOR: [2, (20, 200, 0)],
			GATEWAY: [3, (200, 100, 0)],
			CYBERNETICSCORE: [3, (150, 150, 0)],
			STARGATE: [3, (200,0,0)],
			ROBOTICSFACILITY: [3, (210, 150, 0)],

			VOIDRAY: [2, (255, 100, 0)],
			OBSERVER: [2, (255, 255, 255)]
		}

		for ut in draw_dict:
			for unit in self.units(ut).ready:
				pos = unit.position
				# 记录主基地的位置
				cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[ut][0], draw_dict[ut][1], -1)
		# 主基地名称
		main_base_names = ["nexus", "supplydepot", "hatchery"]
		# 农民名称
		worker_names = ["probe", "scv", "drone"]
		# 记录地方基地的位置
		for enemy_building in self.known_enemy_structures:
			pos = enemy_building.position
			# 不是主基地建筑，画小一些
			if enemy_building.name.lower() not in main_base_names:
				cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 200), -1)
			else:
				cv2.circle(game_data, (int(pos[0]), int(pos[1])), 10, (0, 0, 255), -1)
		# 记录敌方单位
		for enemy_unit in self.known_enemy_units:
			pos = enemy_unit.position
			# 不是农民，画大一些
			if enemy_unit.name.lower() not in worker_names:
				cv2.circle(game_data, (int(pos[0]), int(pos[1])), 2, (20, 0, 200), -1)
			else:
				cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (50, 50, 255), -1)
		# 垂直翻转图像
		flipped = cv2.flip(game_data, 0)
		# 图像缩放
		resized = cv2.resize(flipped, dsize=None, fx = 2, fy = 2)
		cv2.imshow('Intel', resized)
		# 每秒刷新图像
		cv2.waitKey(1)


	'''建造农民'''
	async def build_workers(self):
		# 一个主基地16个农民，大于现有农民数量并且小于最大农民数
		if len(self.units(NEXUS)) * 16 > len(self.units(PROBE)) and len(self.units(PROBE)) < self.MAX_WORKERS:
			# 主基地无队列建造时：
			for i in self.units(NEXUS).ready.idle:
				# 能够买的起农民
				if self.can_afford(PROBE):
					# 训练一个农民
					await self.do(i.train(PROBE))

	# 建造水晶塔
	async def build_pylons(self):
		# 供应人口剩余小于7且水晶塔不是正在建造
		if self.supply_left < 7 and not self.already_pending(PYLON):
			nexuses = self.units(NEXUS).ready
			if nexuses.exists:
				# 能够消费的起
				if self.can_afford(PYLON):
					# 开始建造
					await self.build(PYLON, near=nexuses.first)

	# 建造吸收厂
	async def build_assimilator(self):
		for nexus in self.units(NEXUS).ready:
			#在nexus附件的瓦斯泉上建立吸收厂
			vespenes = self.state.vespene_geyser.closer_than(15.0, nexus)
			# 迭代瓦斯泉
			for vespene in vespenes:
				# 如果没钱就退出
				if not self.can_afford(ASSIMILATOR):
					break
				# 选择该瓦斯泉附近的工人
				worker = self.select_build_worker(vespene.position)
				if worker is None:
					break
				# 如果没有在该瓦斯泉上建立吸收厂就建立
				if not self.units(ASSIMILATOR).closer_than(1, vespene).exists:
					await self.do(worker.build(ASSIMILATOR, vespene))

	# 分基地
	async def expand(self):
		# 动态发展
		if self.units(NEXUS).amount < (self.iteration / self.ITERATION_PER_MINUTE / 2.5) and self.can_afford(NEXUS) and not self.already_pending(NEXUS):
			await self.expand_now()

	

			

	# 建造军事建筑
	async def offensive_force_buildings(self):
		# print(self.iteration / self.ITERATION_PER_MINUTE)
		if self.units(PYLON).ready.exists:
			# 随机取一个水晶塔，在其附近建造
			pylon = self.units(PYLON).ready.random
			if self.units(PYLON).ready.exists:
				# 如果跃迁门存在，则建造控制核心
				if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE).ready.exists:
					if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
						await self.build(CYBERNETICSCORE, near=pylon)
				# 否则就建造跃迁门
				elif len(self.units(GATEWAY)) < 1:
					if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
						await self.build(GATEWAY, near=pylon)
				# 否则就
				if self.units(CYBERNETICSCORE).ready.exists:
					# 建造星门
					if len(self.units(STARGATE)) < (self.iteration / self.ITERATION_PER_MINUTE):
						if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
							await self.build(STARGATE, near=pylon)
					# 建造机械台
					if len(self.units(ROBOTICSFACILITY)) < 1:
						if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
							await self.build(ROBOTICSFACILITY, near=pylon)

	# 造兵
	async def build_offensive_force(self):
		# 跃迁门没有队列时建造跟踪者
		# for gateway in self.units(GATEWAY).ready.idle:
		# 	# 跟踪者小于虚空射线时才建造跟踪者
		# 	if self.units(STALKER).amount <= self.units(VOIDRAY).amount:
		# 		if self.can_afford(STALKER) and self.supply_left > 0:
		# 			await self.do(gateway.train(STALKER))
		# 星门没有队列时建造虚空射线
		for stargate in self.units(STARGATE).ready.idle:
			if self.can_afford(VOIDRAY) and self.supply_left > 0:
				await self.do(stargate.train(VOIDRAY))


	def random_location_variance(self, enemy_start_location):
		x = enemy_start_location[0]
		y = enemy_start_location[1]
		x += ((random.randrange(-20, 20)) / 100) * enemy_start_location[0]
		y += ((random.randrange(-20, 20)) / 100) * enemy_start_location[1]
		if x < 0:
		 	x = 0
		if y < 0:
			y = 0
		if x > self.game_info.map_size[0]:
			x = self.game_info.map_size[0]
		if y > self.game_info.map_size[1]:
			y = self.game_info.map_size[1]
		go_to = position.Point2(position.Pointlike((x, y)))
		return go_to



	# 侦察兵行动
	async def scout(self):
		if len(self.units(OBSERVER)) > 0:
			scout = self.units(OBSERVER)[0]
			if scout.is_idle:
				enemy_location = self.enemy_start_locations[0]
				move_to = self.random_location_variance(enemy_location)
				print(move_to)
				await self.do(scout.move(move_to))
		else:
			for rf in self.units(ROBOTICSFACILITY).ready.idle:
				if self.can_afford(OBSERVER) and self.supply_left > 0:
					await self.do(rf.train(OBSERVER))


	# 发现目标
	def find_target(self, state):
		if len(self.known_enemy_units) > 0:
			return random.choice(self.known_enemy_units)
		elif len(self.known_enemy_structures) > 0:
			return random.choice(self.known_enemy_structures)
		else:
			return self.enemy_start_locations[0]

	# 进攻
	async def attack(self):
		aggressive_units = {STALKER : [15, 5], VOIDRAY : [14, 1]}
		for unit in aggressive_units:
			# 进攻模式
			if self.units(unit).amount > aggressive_units[unit][0]:
				for s in self.units(unit).idle:
					await self.do(s.attack(self.find_target(self.state)))
			# 防守模式
			if self.units(unit).amount > aggressive_units[unit][1] :
				if len(self.known_enemy_units) > 0:
					for s in self.units(unit).idle:
						await self.do(s.attack(random.choice(self.known_enemy_units)))


# 设置开始
run_game(maps.get("AbyssalReefLE"), [
	Bot(Race.Protoss, MyBot()), 
	Computer(Race.Zerg, Difficulty.Hard)
	], realtime = False)